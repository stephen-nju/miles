"""WeightBridge: the explicit train->rollout parameter contract for the FSDP backend.

The training side (stock HF / FSDP) and the SGLang rollout loader do not always agree on
parameter *names and shapes*. Each (model_type) that disagrees needs a transform that rewrites
the streamed ``(name, tensor)`` into what the rollout loader expects — e.g. transformers>=5.6
stores qwen3_moe experts as one batched ``experts.gate_up_proj`` / ``experts.down_proj`` tensor,
but SGLang's qwen3_moe loader expects per-expert ``experts.{i}.{gate,up,down}_proj.weight``.

This module makes that contract a first-class, registered object instead of a hardcoded
``if model_type == ...`` branch. It is the FSDP analogue of the Megatron ``megatron_to_hf``
converter registry: a new arch registers a ``ParamTransform`` rather than editing the sync loop.

A transform is a pair of callables:
  * ``matches(name, param) -> bool`` — does this transform apply to this param?
  * ``expand(name, full) -> Iterable[(name, tensor)]`` — pure tensor logic producing the rollout
    stream from the *materialized* (unsharded, on-device) tensor.
Keeping ``expand`` pure (no DTensor/device handling) makes every transform unit-testable on CPU;
the device/DTensor materialization stays in the one place that owns it (update_weight_utils).
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch


class ParamTransform(NamedTuple):
    """A registered train->rollout transform.

    ``matches(name, param) -> bool`` selects which params this transform applies to;
    ``expand(name, full) -> Iterable[(name, tensor)]`` rewrites the materialized tensor
    into the rollout stream.
    """

    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]]


# model_type -> registered transforms, tried in registration order
_REGISTRY: dict[str, list[ParamTransform]] = {}


def register_param_transform(model_type: str, matches: Callable, expand: Callable) -> None:
    _REGISTRY.setdefault(model_type, []).append(ParamTransform(matches, expand))


def get_param_transform(name: str, param, model_type: str):
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for transform in _REGISTRY.get(model_type, ()):
        if transform.matches(name, param):
            return transform.expand
    return None


# ------------------------------------------------------------------------------------------------
# qwen3_moe: split transformers>=5.6 batched experts back into the per-expert names SGLang expects.
# ------------------------------------------------------------------------------------------------
def _qwen3_moe_matches(name: str, param) -> bool:
    return getattr(param, "dim", lambda: 0)() == 3 and (
        name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    )


def _qwen3_moe_expand(name: str, full: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
    """experts.gate_up_proj [E,2I,H] -> experts.{i}.{gate,up}_proj.weight; down_proj [E,H,I] -> per-expert."""
    prefix = name.rsplit(".", 1)[0]  # ...mlp.experts
    num_experts = full.shape[0]
    if name.endswith(".gate_up_proj"):
        half = full.shape[1] // 2  # fused rows are [gate | up]
        for i in range(num_experts):
            yield f"{prefix}.{i}.gate_proj.weight", full[i, :half, :].contiguous()
            yield f"{prefix}.{i}.up_proj.weight", full[i, half:, :].contiguous()
    else:  # .down_proj
        for i in range(num_experts):
            yield f"{prefix}.{i}.down_proj.weight", full[i].contiguous()


# ``_qwen3_moe_matches`` + ``_qwen3_moe_expand`` are a reusable building block: every arch with the
# transformers>=5.6 batched-expert layout (qwen3_moe, glm4_moe_lite, ...) registers them in its spec
# (see adaptations/specs/). The registrations live there, not here, so this module stays pure mechanism.
