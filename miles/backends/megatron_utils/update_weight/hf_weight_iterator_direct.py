import dataclasses
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf, get_atomic_update_groups
from ..sglang import monkey_patch_torch_reductions
from .common import all_gather_params_async, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase


@dataclasses.dataclass(frozen=True)
class _UpdateUnit:
    params: tuple[ParamInfo, ...]
    size: int


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(
            self.args, self.model, self.model_name
        )

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            megatron_full_params = _get_megatron_full_params(
                self.args, megatron_local_param_infos, megatron_local_weights
            )
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors


def _get_megatron_full_params(
    args: Namespace,
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = get_parallel_state().pp.size
    ep_size = get_parallel_state().ep.size
    rank = dist.get_rank()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(get_parallel_state().pp.group):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=get_parallel_state().pp.group, async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(get_parallel_state().ep.group)
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=get_parallel_state().ep.group, async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(args, list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _get_megatron_local_param_info_buckets(
    args: Namespace, model: Sequence[torch.nn.Module], model_name: str
) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).

    Model-specific atomic update groups are kept in the same bucket because
    some rollout loaders must see related tensors in the same load_weights call.
    """
    param_infos = _get_megatron_local_param_infos(args, model)
    update_units = _get_update_units(model_name, param_infos)
    return _pack_update_units(args, update_units)


def _get_param_full_size(info: ParamInfo) -> int:
    if ".experts." in info.name:
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _get_update_units(model_name: str, param_infos: list[ParamInfo]) -> list[_UpdateUnit]:
    by_name = {info.name: info for info in param_infos}
    position = {info.name: i for i, info in enumerate(param_infos)}
    grouped_names: set[str] = set()
    group_keys: set[str] = set()
    ordered_units: list[tuple[int, _UpdateUnit]] = []

    for group in get_atomic_update_groups(model_name, param_infos):
        key = group.key
        names = group.names
        if key in group_keys:
            raise RuntimeError(f"Duplicate atomic update group: {key}")
        if not names:
            raise RuntimeError(f"Atomic update group {key} has no params")
        group_keys.add(key)

        params = []
        for name in names:
            if name in grouped_names:
                raise RuntimeError(f"Param {name} appears in multiple atomic update groups")
            if name not in by_name:
                raise RuntimeError(f"Atomic update group {key} references unknown param {name}")
            grouped_names.add(name)
            params.append(by_name[name])

        ordered_units.append(
            (
                min(position[name] for name in names),
                _UpdateUnit(params=tuple(params), size=sum(_get_param_full_size(param) for param in params)),
            )
        )

    for info in param_infos:
        if info.name not in grouped_names:
            ordered_units.append((position[info.name], _UpdateUnit(params=(info,), size=_get_param_full_size(info))))

    return [unit for _position, unit in sorted(ordered_units, key=lambda item: item[0])]


def _pack_update_units(args: Namespace, update_units: list[_UpdateUnit]) -> list[list[ParamInfo]]:
    param_info_buckets: list[list[ParamInfo]] = [[]]
    buffer_size = 0

    for unit in update_units:
        if buffer_size + unit.size > args.update_weight_buffer_size and param_info_buckets[-1]:
            param_info_buckets.append([])
            buffer_size = 0
        param_info_buckets[-1].extend(unit.params)
        buffer_size += unit.size

    return param_info_buckets


def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = get_parallel_state().pp.size
    ep_size = get_parallel_state().ep.size

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().pp.group
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    assert args.mtp_num_layers is not None
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().ep.group
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    return param_infos
