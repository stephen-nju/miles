import io
import logging
from collections.abc import Sequence
from datetime import timedelta

import torch

try:
    from torchft.checkpointing.pg_transport import PGTransport
except ImportError:
    PGTransport = None

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

# Must accommodate receiver's model init time (can take minutes for large models)
_DEFAULT_TIMEOUT = timedelta(seconds=600)


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> None:
    """Send in-memory checkpoint to a destination cell via torchft PGTransport."""
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )

    payload = _serialize_for_transport(state_dict, iteration)
    transport = _create_transport(indep_dp, timeout)
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=0,
        state_dict=payload,
        timeout=timeout,
    )
    transport.disallow_checkpoint()
    logger.info(f"Sent checkpoint (iteration={iteration}) to alive_rank={dst_rank}")


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> InMemoryCheckpointManager:
    """Receive checkpoint from a healthy cell via torchft PGTransport."""
    transport = _create_transport(indep_dp, timeout)
    payload = transport.recv_checkpoint(
        src_rank=src_rank,
        metadata=transport.metadata(),
        step=0,
        timeout=timeout,
    )
    iteration, state_dict = _deserialize_from_transport(payload)
    logger.info(f"Received checkpoint (iteration={iteration}) from alive_rank={src_rank}")

    manager = InMemoryCheckpointManager()
    manager.save(state_dict, iteration=iteration)
    return manager


def _serialize_for_transport(state_dict: object, iteration: int) -> dict[str, torch.Tensor]:
    """Serialize MCoreTensorAwareStateDict to a single tensor for PGTransport.

    PGTransport uses tree_flatten_with_path to split tensor vs non-tensor leaves.
    MCoreTensorAwareStateDict is not pytree-registered, so it gets pickled as
    metadata — triggering torch.save on every tensor (slow, CPU-only).

    Fix: torch.save the whole thing to bytes, wrap as a uint8 tensor.
    PGTransport sees one tensor → sends via NCCL P2P (fast).
    """
    buf = io.BytesIO()
    torch.save({"iteration": iteration, "state_dict": state_dict}, buf)
    return {"ckpt_bytes": torch.frombuffer(buf.getvalue(), dtype=torch.uint8)}


def _deserialize_from_transport(payload: dict[str, torch.Tensor]) -> tuple[int, object]:
    """Inverse of _serialize_for_transport."""
    ckpt_bytes = payload["ckpt_bytes"].cpu().numpy().tobytes()
    data = torch.load(io.BytesIO(ckpt_bytes), weights_only=False)
    return data["iteration"], data["state_dict"]


def _create_transport(indep_dp: GroupInfo, timeout: timedelta) -> PGTransport:
    return PGTransport(
        pg=indep_dp.group,
        timeout=timeout,
        device=torch.device("cuda"),
    )
