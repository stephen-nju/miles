"""torchft PG abort() behavior matrix: under which peer states does abort hang?

NOT a CI test. Minimal reproduction of the r2 reconfigure hang (see
agent-context 2026-06-07-indep-dp-pg-graveyard-explainer) plus boundary
conditions, to answer: why do torchft/torchtitan deployments not hit the hang
miles hit, and which teardown path is actually safe.

Topology (4 GPUs, mirrors two miles cells of 2 ranks each):
    cell A = {a0, a1}, cell B = {b0, b1}
    per-cell torchft NCCL PG:    (a0,a1) and (b0,b1)
    cross-cell torchft NCCL+Gloo pair PGs: (a0,b0) and (a1,b1)
All PGs are warmed up with one allreduce, then left idle.

Experiments (each on a fresh actor set):
    torchft_form  kill a0 AND a1 (the whole peer cell dies, mirroring the
                  replica-granularity failures torchft deployments produce)
                  -> b1 aborts its cross pair. Expect: fast.
    r2_form       kill a0 only; a1 stays alive but wedged (sleeping, comms
                  idle) -> b1 aborts its cross pair. Expect (r2): hang; then
                  kill a1 and expect the abort to unblock.
    locate_nccl   r2_form but b1 aborts ONLY the cross NCCL PG.
    locate_gloo   r2_form but b1 aborts ONLY the cross Gloo PG.
    gpu_spin      r2_form but a1 spins a GPU kernel while wedged (closer to
                  the real wedged-in-collective state).

Usage:
    python tests/e2e/external/torchft_abort_matrix_experiment.py
    python tests/e2e/external/torchft_abort_matrix_experiment.py --only r2_form
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_HANG_VERDICT_TIMEOUT_S = 45.0
_SETTLE_S = 3.0


@ray.remote(num_gpus=1)
class _MatrixWorker:
    """One rank: holds a cell-internal PG and a cross-cell pair PG (NCCL + Gloo)."""

    def setup(self, *, store_base: str, name: str, timeout_s: float) -> dict:
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._name = name
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        cell = name[0]  # "a" / "b"
        cell_rank = int(name[1])  # 0 / 1
        pair_idx = cell_rank  # pair0 = (a0,b0), pair1 = (a1,b1)
        pair_rank = 0 if cell == "a" else 1

        def _mk(pg_cls: type, store_suffix: str, rank: int) -> object:
            pg = pg_cls(timeout=timedelta(seconds=timeout_s))
            pg.configure(
                store_addr=f"{store_base}/{store_suffix}",
                replica_id=name,
                rank=rank,
                world_size=2,
                quorum_id=0,
            )
            return pg

        self._cell_pg = _mk(ProcessGroupNCCL, f"cell_{cell}", cell_rank)
        self._cross_pg = _mk(ProcessGroupNCCL, f"pair{pair_idx}_nccl", pair_rank)
        self._cross_gloo = _mk(ProcessGroupGloo, f"pair{pair_idx}_gloo", pair_rank)

        # Warm every comm up with one allreduce, then leave it idle (mirrors a
        # quorum that worked in the previous step and is torn down later).
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        for pg, dev in [(self._cell_pg, self._device), (self._cross_pg, self._device), (self._cross_gloo, "cpu")]:
            t = torch.ones(8, device=dev)
            assert pg.allreduce([t], opts).wait(), f"{name}: warmup failed"

        return {"name": name, "nccl": torch.cuda.nccl.version()}

    def die(self) -> None:
        os._exit(1)

    def wedge_sleep(self) -> None:
        """Stay alive but never respond (idle comms, blocked main thread)."""
        time.sleep(1_000_000)

    def wedge_gpu_spin(self) -> None:
        """Stay alive with a busy GPU (closer to wedged-in-collective)."""
        import torch

        x = torch.randn(4096, 4096, device=self._device)
        while True:
            x = x @ x
            x = x / x.norm()

    def abort_cross(self, *, target: str) -> dict:
        """Abort this rank's cross-cell PG(s) like reconfigure_indep_dp_group does; time it."""
        groups = {
            "nccl": [self._cross_pg],
            "gloo": [self._cross_gloo],
            "both": [self._cross_pg, self._cross_gloo],
        }[target]

        timings: list[tuple[str, float]] = []
        start_all = time.monotonic()
        for pg in groups:
            start = time.monotonic()
            pg.abort(errored=False)
            timings.append((type(pg).__name__, round(time.monotonic() - start, 3)))
        return {
            "name": self._name,
            "target": target,
            "total_s": round(time.monotonic() - start_all, 3),
            "per_pg_s": timings,
        }


def _spawn_cells(*, store_base: str, timeout_s: float) -> dict:
    workers = {name: _MatrixWorker.remote() for name in ("a0", "a1", "b0", "b1")}
    infos = ray.get(
        [w.setup.remote(store_base=store_base, name=name, timeout_s=timeout_s) for name, w in workers.items()],
        timeout=120,
    )
    print(f"  setup: {infos}")
    return workers


def _fire_and_forget(ref: object) -> None:
    del ref


def _kill_all(workers: dict) -> None:
    for w in workers.values():
        try:
            ray.kill(w, no_restart=True)
        except Exception:  # noqa: BLE001 - idempotent cleanup
            pass


def _await_abort(ref: object, *, workers: dict, expect: str) -> str:
    """Wait for the abort call; on hang, kill the wedged peer and wait again."""
    try:
        out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
        return f"returned in {out['total_s']}s per_pg={out['per_pg_s']} (expect={expect})"
    except ray.exceptions.GetTimeoutError:
        pass

    print(f"  abort STUCK for >{_HANG_VERDICT_TIMEOUT_S}s -> killing wedged peer a1 to test unblock")
    ray.kill(workers["a1"], no_restart=True)
    try:
        out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
        return f"HUNG then UNBLOCKED by peer kill: {out['total_s']}s total (expect={expect})"
    except ray.exceptions.GetTimeoutError:
        return f"HUNG and STILL STUCK {_HANG_VERDICT_TIMEOUT_S}s after peer kill (expect={expect})"


def _run_experiment(exp: str, *, store_base: str, timeout_s: float) -> str:
    print(f"== experiment {exp} ==")
    workers = _spawn_cells(store_base=f"{store_base}/{exp}", timeout_s=timeout_s)

    try:
        if exp == "torchft_form":
            _fire_and_forget(workers["a0"].die.remote())
            _fire_and_forget(workers["a1"].die.remote())
            time.sleep(_SETTLE_S)
            ref = workers["b1"].abort_cross.remote(target="both")
            return _await_abort(ref, workers=workers, expect="fast")

        wedge_method = "wedge_gpu_spin" if exp == "gpu_spin" else "wedge_sleep"
        target = {"locate_nccl": "nccl", "locate_gloo": "gloo"}.get(exp, "both")

        _fire_and_forget(getattr(workers["a1"], wedge_method).remote())
        time.sleep(_SETTLE_S)  # a1 occupies its single actor thread = wedged
        _fire_and_forget(workers["a0"].die.remote())
        time.sleep(_SETTLE_S)
        ref = workers["b1"].abort_cross.remote(target=target)
        return _await_abort(ref, workers=workers, expect="hang? (r2 says yes for some target)")
    finally:
        _kill_all(workers)


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout")] = 20.0,
    only: Annotated[str | None, typer.Option(help="run a single experiment by name")] = None,
) -> None:
    """Run the abort-behavior matrix; prints one RESULT line per experiment."""
    ray.init(ignore_reinit_error=True)

    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_base = f"localhost:{store.port}/abort_matrix"

    experiments = ["torchft_form", "r2_form", "locate_nccl", "locate_gloo", "gpu_spin"]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for exp in experiments:
        results[exp] = _run_experiment(exp, store_base=store_base, timeout_s=timeout_s)
        print(f"RESULT {exp}: {results[exp]}\n")
        time.sleep(2)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")

    del store


if __name__ == "__main__":
    typer.run(main)
