# Fault Tolerance E2E Tests

## Test Overview

| Test | Type | What it verifies |
|------|------|-----------------|
| `test_trainer_ft_no_failure.py` | Comparison | indep_dp matches normal DP when no faults |
| `test_trainer_ft_with_failure.py` | Comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `test_trainer_ft_deterministic.py` | Comparison | indep_dp matches normal DP with stop+start healing (no missed steps) |
| `test_ft_random.py` | Non-comparison | System survives random crashes without hanging |

## Mode Variants

Each test runs with `--mode`:

All modes are **disaggregated** (training and rollout on separate nodes). Modes without rollout use debug rollout data.

| Mode | Nodes | DP cells | Batch | Parallelism | Rollout | Coverage |
|------|-------|----------|-------|-------------|---------|----------|
| `dp2_cp2_tp2_ep2` | 1 | 2 | 3 | CP2 TP2 EP2 | debug data | TP + EP |
| `dp2_cp2_pp2` | 1 | 2 | 3 | CP2 PP2 | debug data | PP |
| `dp4_cp2` | 1 | 4 | 5 | CP2 | debug data | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 1 | 2 | 3 | CP2 | 4 engines × 1 GPU | Real weight update path |
| `6node_dp4_cp2_tp2_pp2_ep2_etp2` | 4+2 | 4 | 5 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | Large-scale, all parallelism |

Batch sizes are deliberately **not** divisible by num_cells to test uneven sample distribution across replicas (e.g. DP4 + batch 5 → 2,1,1,1).

## Running

**Required**: `MILES_SCRIPT_OUTPUT_DIR` environment variable must be set. Dump files are written to `$MILES_SCRIPT_OUTPUT_DIR/dumps/<test_name>/`.

### Comparison tests (`test_trainer_ft_no_failure.py`, `test_trainer_ft_with_failure.py`, `test_trainer_ft_deterministic.py`)

These compare baseline (normal DP) against target (indep_dp). They support 4 subcommands:

- `run` — full pipeline: prepare + baseline + target + compare
- `baseline` / `target` — run one side independently (useful for debugging)
- `compare` — re-run comparison on existing dumps (no GPU needed)

```bash
# Full pipeline (CI)
python tests/e2e/ft/test_trainer_ft_no_failure.py run --mode dp2_cp2_tp2_ep2

# Step by step (debugging)
python tests/e2e/ft/test_trainer_ft_no_failure.py baseline --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py target   --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py compare  --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft

# With-failure has phases (phase_a saves ckpt, phase_b resumes + injects fault)
python tests/e2e/ft/test_trainer_ft_with_failure.py run --mode dp4_cp2

# Deterministic healing (designed for large-scale disagg)
python tests/e2e/ft/test_trainer_ft_deterministic.py run --mode 6node_dp4_cp2_tp2_pp2_ep2_etp2
```

### Non-comparison tests (`test_ft_random.py`)

Single `run` subcommand. No baseline — just verifies the system doesn't crash.

```bash
python tests/e2e/ft/test_ft_random.py run --mode dp4_cp2 --seed 42 --num-steps 50
```

## Debug Rollout Data

Modes without rollout engines (`has_rollout == False`) use pre-recorded rollout data via `--load-debug-rollout-data --debug-train-only`, skipping real rollout generation.

`conftest_ft/execution.py` `prepare()` downloads the data via `U.hf_download_dataset()`.

### How to regenerate

The debug rollout data **must** be generated using the 5-layer model (not the full model).
Using the full model produces `rollout_log_probs` incompatible with the 5-layer training model,
causing NaN gradients in GRPO training.

```bash
# Step 1: Generate rollout data (5-layer model + real sglang rollout, no dumper)
python tests/e2e/ft/test_trainer_ft_no_failure.py generate-data \
    --mode dp2_cp2_real_rollout --num-steps 12 --output-dir /tmp/gen_rollout

# Step 2: Locate the generated rollout data
ls /tmp/gen_rollout/rollout_data/

# Step 3: Upload to HF
huggingface-cli upload --repo-type dataset fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer \
    /tmp/gen_rollout/rollout_data/
```

---

## Test Definitions

### `test_trainer_ft_no_failure.py`

Comparison test. Verifies indep_dp produces the same results as normal DP when no faults occur.

```
Type: comparison (baseline=normal DP, target=indep_dp)
Steps: 2

1. Baseline: run normal DP training with debug rollout data
2. Target: run indep_dp training with the same data
3. Compare:
   - Tensor-level: compare_dumps (weights, grads via dumper & sglang comparator)
   - Metric-level: compare_metrics (MetricEvent, requires train/grad_norm and train/loss)

Note: results are roughly equal, not bitwise — allreduce kernel ordering differs.
```

### `test_trainer_ft_with_failure.py`

Multi-phase comparison test. Verifies indep_dp matches normal DP after fault + checkpoint resume.

```
Type: comparison, multi-phase (phase_a + phase_b)
Phase A steps: 1, Phase B steps: 4, rtol: 5e-2

Phase A (both baseline and target):
  1. Run 1 step of training
  2. Save checkpoint (--save-interval 1)

Phase B — baseline:
  1. Resume from phase_a checkpoint
  2. Run 4 normal steps

Phase B — target:
  1. Resume from phase_a checkpoint
  2. Rollout 1: N cells normal
  3. Rollout 2, attempt 0: crash_before_allreduce on last cell rank 0
     → os._exit(1) → allreduce timeout → should_commit=false → retry
  4. Rollout 2, attempt 1: _refresh_cells() reconfigure → N-1 cells → commit
  5. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last)
  6. Rollout 3: _refresh_cells() healing → N cells
  7. Rollout 4: N cells stable

Compare: phase_b dumps and metrics (baseline vs target, rtol=5e-2 for accumulated error).

Fault injection via --ci-ft-test-actions JSON (data-driven, executed by RayTrainGroup).
The JSON `at_rollout` field specifies which rollout_id triggers the action (replaces old `after_step`).
The `attempt` field (for actor-level actions like `crash_before_allreduce`) specifies which retry attempt to match.
```

### `test_trainer_ft_deterministic.py`

Multi-phase comparison test. Verifies healing state transfer is bitwise correct.

```
Type: comparison, multi-phase (phase_a + phase_b)
Phase A steps: 1, Phase B steps: 3, rtol: 1e-2 (3e-2 / atol 2e-8 for real-rollout modes,
which restart SGLang on stop+start_cell and cannot be bitwise reproducible)

Phase A: same as with_failure (1 step + save ckpt).

Phase B — target timeline:
  1. Rollout 1, 2: all N cells normal (2 good steps, accumulate meaningful state)
  2. After rollout 2: stop_cell_at_end(last) + start_cell_at_end(last) — trigger healing
  3. Rollout 3: healing at start (recv_ckpt from cell_0), then normal execution

Both baseline and target use --deterministic-mode + env vars (NCCL_ALGO=Ring,
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, CUBLAS_WORKSPACE_CONFIG=:4096:8) for
bitwise reproducibility.

Bitwise verification: --use-fault-tolerance --ft-components train auto-enables
--save-local-weight-checksum and --enable-event-analyzer. The event_analyzer
cross_replica_weight_checksum rule checks cell-to-cell bitwise equality after healing.
```

### `test_ft_random.py`

Non-comparison soak test. Verifies the system survives random crashes without hanging.

```
Type: non-comparison (no baseline, no compare)
Steps: 30 (default), configurable via --num-steps

Architecture (external fault injection, not inside training loop):
  1. Start training with indep_dp + control server (port 18080) + mini FT controller
  2. Start a background daemon thread that:
     a. Sleeps a random interval (exponential distribution, mean ~15s / crash_probability)
     b. GET /api/v1/cells — discover alive cells
     c. Pick a random alive cell (skip if <=1 alive)
     d. POST /api/v1/cells/{name}/inject-fault with random failure mode
     e. Repeat until training finishes
  3. The actor's inject_fault() runs in a dedicated ray concurrency group thread
     and kills the process immediately (SIGKILL, os._exit, or segfault)
  4. Health checker detects dead actor via heartbeat timeout
  5. Mini FT controller auto-recovers (suspend → resume)
  6. Verify: training completes, no hangs, prod assertions pass

CLI options: --seed (default 42), --num-steps (default 30), --crash-probability (default 0.1)
```
