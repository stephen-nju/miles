from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-c-4-gpu-h200", labels=[])

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.utils.det_process_group import DetProcessGroup

_WORLD_SIZE = 4
_NUMEL = 1_000_000
_SEED = 1234


def _order_sensitive_input(rank: int) -> torch.Tensor:
    """Per-rank input whose cross-rank sum catastrophically cancels (~1e-4 from +-0.5)."""
    shared = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(_SEED), dtype=torch.float32)
    own = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(_SEED + 1 + rank), dtype=torch.float32)
    sign = -1.0 if rank % 2 else 1.0
    return (sign * 0.5 * shared + 1e-4 * own).cuda()


def _manual_tree_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


def _check_deterministic_allreduce(rank: int, det1: DetProcessGroup, det2: DetProcessGroup) -> torch.Tensor:
    x = _order_sensitive_input(rank)
    gathered = [torch.empty_like(x) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, x)
    expected = _manual_tree_sum(gathered)

    direct = x.clone()
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    assert det1.allreduce([direct], opts).wait()
    assert torch.equal(direct, expected), "direct det allreduce != fixed tree fold"

    dispatched = x.clone()
    dist.all_reduce(dispatched, op=dist.ReduceOp.SUM, group=det1)
    assert torch.equal(dispatched, expected), "dist.all_reduce(group=det) bypassed the override"

    other_comm = x.clone()
    dist.all_reduce(other_comm, op=dist.ReduceOp.SUM, group=det2)
    assert torch.equal(other_comm, expected), "det allreduce differs across communicator instances"

    averaged = x.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.AVG, group=det1)
    assert torch.equal(averaged, expected / _WORLD_SIZE), "det AVG != det SUM / world"

    max_det = x.clone()
    dist.all_reduce(max_det, op=dist.ReduceOp.MAX, group=det1)
    max_native = x.clone()
    dist.all_reduce(max_native, op=dist.ReduceOp.MAX)
    assert torch.equal(max_det, max_native), "MAX should delegate to the native implementation"

    return expected


def _check_deterministic_reduce_scatter(rank: int, det1: DetProcessGroup, expected: torch.Tensor) -> None:
    x = _order_sensitive_input(rank)
    shard_numel = _NUMEL // _WORLD_SIZE
    expected_shard = expected[rank * shard_numel : (rank + 1) * shard_numel]

    out = torch.empty(shard_numel, device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, group=det1)
    assert torch.equal(out, expected_shard), "det reduce_scatter != its slice of the det allreduce"

    # Megatron distributed-optimizer style: the output shard is a view of the input.
    buf = x.clone()
    shard_view = buf.view(_WORLD_SIZE, -1)[rank]
    dist.reduce_scatter_tensor(shard_view, buf, op=dist.ReduceOp.SUM, group=det1)
    assert torch.equal(shard_view, expected_shard.view(shard_view.shape)), "aliased reduce_scatter broken"


def _check_non_contiguous_allreduce(rank: int, det1: DetProcessGroup) -> None:
    base = torch.randn(64, 64, generator=torch.Generator().manual_seed(_SEED + 100 + rank)).cuda()
    non_contiguous = base.t()
    assert not non_contiguous.is_contiguous()

    gathered = [torch.empty_like(base) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, base)
    expected = _manual_tree_sum([g.t().contiguous() for g in gathered])

    dist.all_reduce(non_contiguous, op=dist.ReduceOp.SUM, group=det1)
    assert torch.equal(non_contiguous.contiguous(), expected), "non-contiguous det allreduce broken"


def _check_delegation(rank: int, det1: DetProcessGroup) -> None:
    broadcasted = torch.full((8,), float(rank), device="cuda")
    dist.broadcast(broadcasted, src=0, group=det1)
    assert torch.equal(broadcasted, torch.zeros(8, device="cuda")), "broadcast delegation broken"

    piece = torch.full((4,), float(rank), device="cuda")
    pieces = [torch.empty_like(piece) for _ in range(_WORLD_SIZE)]
    dist.all_gather(pieces, piece, group=det1)
    for source_rank, gathered_piece in enumerate(pieces):
        assert torch.equal(gathered_piece, torch.full((4,), float(source_rank), device="cuda"))

    # Pairwise exchange (0<->1, 2<->3) through the wrapper's send/recv delegation.
    peer = rank + 1 if rank % 2 == 0 else rank - 1
    outgoing = torch.full((4,), float(rank), device="cuda")
    incoming = torch.empty(4, device="cuda")
    if rank % 2 == 0:
        dist.send(outgoing, dst=peer, group=det1)
        dist.recv(incoming, src=peer, group=det1)
    else:
        dist.recv(incoming, src=peer, group=det1)
        dist.send(outgoing, dst=peer, group=det1)
    assert torch.equal(incoming, torch.full((4,), float(peer), device="cuda")), "send/recv delegation broken"

    dist.barrier(group=det1)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    inner1 = dist.new_group(list(range(world_size)), backend="nccl")
    inner2 = dist.new_group(list(range(world_size)), backend="nccl")
    det1 = DetProcessGroup(inner1)
    det2 = DetProcessGroup(inner2)

    expected = _check_deterministic_allreduce(rank, det1, det2)
    _check_deterministic_reduce_scatter(rank, det1, expected)
    _check_non_contiguous_allreduce(rank, det1)
    _check_delegation(rank, det1)

    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def test_det_process_group_multi_gpu():
    """DetProcessGroup gives bitwise fixed-order SUM/AVG reductions and faithful delegation."""
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")

    mp.spawn(_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
