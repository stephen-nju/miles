# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Add an OPT-IN multi-process mode to the Miles session server so that CPU-bound response handling (chiefly `json.loads`/validation of large R3 `routed_experts` bodies) can run in parallel across OS processes and escape the GIL, instead of contending on a single process's GIL as the current bounded thread-pool does.

The architecture is a **thin router process** fronting **N headless worker processes**:

- The router is the only client-facing HTTP listener. For each request it extracts `session_id` from the path, selects `worker_idx = stable_hash(session_id) % n_worker`, forwards the request to that worker over **IPC (not a second HTTP hop)**, awaits the worker's result over IPC, and writes the HTTP response. The router holds no session state and never `json.loads` the chat response body.
- Each worker runs its own asyncio event loop and the existing session logic in-process (`SessionRegistry` / `LinearTrajectory` / per-session in-flight gate). The worker makes the upstream inference HTTP call itself (`backend_url` unchanged) and stores the full record (including R3) locally.
- Sessions are nearly standalone (no cross-session shared state); sticky-by-hash guarantees a session always returns to the same worker, so IPC moves only this-turn's request/response payload, never accumulated session state.

`--session-server-workers=1` (default) preserves today's exact single-process path (no router, no IPC), making the feature fully opt-in and backward compatible.

Resolved decisions: DEC-0 (branch from `9cf2a0384`; re-establish in-flight gate + strict validation, drop thread pool); DEC-1=B (strip R3 from client chat response, keep in worker record, serve via GET); DEC-2 (worker death → whole session server fail-fast); DEC-3 firm (keep 32-hex id, stable hash); DEC-4 (success = directional parallel-parse win; delta-R3 deferred); DEC-5 (IPC transport = allowed choice gated by AC-5/AC-6).

Branching: start from commit `9cf2a0384`; branch `refactor/multi-process-session-server`; do NOT build on `refactor/session-server-concurrency`.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

Following TDD philosophy, each criterion includes positive and negative tests. Behavioral-equivalence tests should be parameterized to run against BOTH `--session-server-workers=1` (single-process adapter) and `--session-server-workers=N` (router + IPC).

- AC-1: Opt-in and backward compatibility. With `--session-server-workers` unset or `=1`, the session server behaves byte-for-byte as the single-process server and spawns no router/extra worker processes.
  - Positive: workers=1 → existing public-HTTP suites (`tests/fast/router/test_sessions.py`, `test_session_race_conditions.py`, `test_session_pretokenized_e2e.py`) pass unchanged; launch produces exactly one session-server process.
  - Negative: workers=1 spawning a router/extra worker fails the process-topology assertion; any altered single-process behavior fails the unchanged suites.

- AC-2: Create-path and sticky routing consistency. With workers=N, a session created via `POST /sessions` is owned by exactly one worker; every `session_id`-keyed endpoint routes to that same worker.
  - Positive: create → chat → GET → DELETE operate on the same worker's state; GET returns the chat's record; catch-all `/sessions/{id}/{path}` (e.g. `/abort_request`) routes to the owner and proxies WITHOUT a session-existence 404 check.
  - Negative: chat/GET/DELETE for an unowned `session_id` → 404 `SessionNotFoundError` (no silent create on an arbitrary worker); catch-all returning 404 for unknown id fails.

- AC-3: Stable, 32-hex session id (firm). The id stays 32-char lowercase-hex; routing uses a process-stable hash independent of `PYTHONHASHSEED`.
  - Positive: `len(session_id)==32` and matches `[0-9a-f]{32}`; same id → same `worker_idx` across separate processes / interpreter restarts.
  - Negative: a non-32-hex id fails the format assertion; a routing function built on Python builtin `hash()` (per-process salted) that remaps an id across a process restart fails the cross-process determinism test.

- AC-4: Behavioral equivalence of the session core across transports. The decoupled core handlers produce identical session-state and error semantics whether driven via single-process FastAPI (workers=1) or via router+IPC (workers=N).
  - Positive (race + pretokenized suites at workers=N): in-flight gate → concurrent second same-session chat returns 409 and does NOT reach the backend; different sessions run concurrently; closing-404 beats busy-409; slot released on every error path; non-200 upstream passes through unrecorded; invalid-200 (missing meta_info / non-numeric logprob / non-int token id) → 502 with nothing committed; TITO prefix invariants hold; ≤1-step rollback; `append_record` R3 pruning identical; framing headers stripped.
  - Negative: a worker mapping `SessionError` to the wrong status fails the 409 test; recording a non-200 or committing on invalid-200 fails; failing to release the in-flight slot on any error path fails.

- AC-5: Worker concurrency model. Within a worker, different sessions' upstream inference calls overlap; CPU parse/validate is bounded; per session there is still exactly one in-flight chat.
  - Positive: K concurrent chats on K distinct sessions on one worker (upstream latency L) complete in ≈ L (overlapped), not ≈ K·L; a DELETE/GET for one session is serviced while another session's chat awaits its upstream call.
  - Negative: a strictly-serial worker completing the K concurrent distinct-session chats in ≈ K·L fails; unbounded in-process parse concurrency reproducing the 16-thread GIL-contention curve fails.

- AC-6: The large-response path does not relocate the bottleneck to the router (DEC-1=B). R3 stripped from the client chat response and kept server-side; the router never parses the chat response body; router memory bounded under concurrency.
  - Positive: with R3 enabled, the client chat response has the assistant message + small meta but NOT `routed_experts`/`indexer_topk`; `GET /sessions/{id}` records still contain R3 for retained turns and `compute_samples_from_openai_records` reconstructs samples (incl. routed_experts) unchanged; the router never `json.loads` a chat response body; on the production-shaped benchmark router RSS does not grow ~`concurrency × body_size`; workers=N improves aggregate parse throughput / tail latency vs workers=1 and vs the thread-pool path (directional; DEC-4).
  - Negative: a client chat response still including the full `routed_experts` blob fails; a router that `json.loads`/re-serializes the full response dict, or whose RSS scales linearly with `concurrency × R3_size` to OOM risk, fails; breaking GET-records R3 retrieval fails.

- AC-7: Lifecycle, health, and crash semantics. The launcher starts a router + N workers under a non-daemon supervisor; readiness waits for every worker's tokenizer/TITO init; the router `/health` reflects worker liveness; shutdown leaves no orphans; any worker death fails the whole session server fast (DEC-2).
  - Positive: startup blocks until all workers ready; `OpenAIEndpointTracer.create()`'s `/health` (stable `session_server_instance_id` shape) succeeds against the router; SIGTERM terminates router + all workers with no orphans; killing one worker → fail fast (no silent empty-worker restart returning fresh-state 200s); a real client disconnect mid-chat does not leak the worker slot or a router-side pending reply.
  - Negative: orphaned worker processes after shutdown fails; a dead shard silently replaced by a fresh empty worker returning 200 for a pre-existing session fails; a leaked in-flight slot / pending IPC reply after disconnect fails.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Completed IMMUTABLE ACs from 3 → 7 (added AC-4..AC-7) | setup script truncated the goal-tracker at AC-3; AC-4/AC-6/AC-7 are core (equivalence, router/R3 win, lifecycle) and must be tracked. Round 0 is the only round IMMUTABLE may change. | none (faithful to plan ACs) |
| 0 | R3 strip is UNIFORM (workers=1 AND workers>1), per user decision after Codex flagged worker-count-dependent API as a semantic smell. | Avoids the chat-response shape depending on worker count (custom agents reading R3 would break only at scale); matches the plan's Behavior-change note (R3 removal = the one intended change vs baseline). No consumer reads R3 from the chat response (agent reads choices[0].message; R3 via GET-records). | AC-1 now means "byte-for-byte EXCEPT the uniform R3 strip"; AC-6 strip applies in both modes; the meta_info passthrough test is updated to expect R3 absent in the client response but present in GET-records. |
| 0 | task13 run as a PRE-implementation adversarial design review (vs post-impl in the plan). | Front-loading the review on the highest-risk milestone (IPC/concurrency) catches design flaws before code is written; far higher value than reviewing after. Findings captured as binding requirements in `m3-design-contract.md`. | strengthens AC-5/AC-6/AC-7 (9 must-fixes folded into the M3 build spec). |
| 0 | task12 scope refined (user request): port `bench_session_server_overhead.py` from `refactor/session-server-concurrency` (it is ABSENT at the 9cf2a0384 base), add a multi-CPU-worker input, and report OLD-vs-NEW latency + throughput at p50/p95/**p99** (was p50-p95) + total wall time + significance. | User explicitly asked to reuse the richer concurrency-branch bench and quantify the latency/throughput/wall-time deltas across percentiles. | AC-6: benchmark made concrete; adds p99; comparison baseline includes the thread-pool path (which exists only on the abandoned branch — the ported bench can drive it). |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1: confirm branch from 9cf2a0384 + document baseline delta vs HEAD | AC-1, AC-4 | done | coding | claude | Branch cut + checkout already done pre-loop; baseline confirmed (no chat_inflight at 9cf2a0384). Document delta in round summary. |
| task2: re-establish per-session in-flight gate (409) + strict upstream-response validation (502), WITHOUT the CPU thread pool | AC-4 | done (pending verification) | coding | claude | M0 commit `f2b4a5e39`: manual derivation, inline (no cpu_executor/ThreadPoolExecutor); 42 tests pass. |
| task3: extract transport-neutral core handlers (create/get/delete/chat/proxy) + SessionError→status; keep thin FastAPI adapter + do_proxy facade | AC-1, AC-4 | done (pending verification) | coding | claude | M1 commit `202113e5e`: new `session_core.py` (`SessionCore`/`CoreResponse`/`ProxyRequest`), transport-neutral (framework refs are docstrings only); 42 required + 98 total fast/router tests pass. |
| task4: add `--session-server-workers` (default 1); wire through router_manager/run_session_server; workers=1 short-circuits | AC-1 | done (pending verification) | coding | claude | M2 commit `fffa0a6ad`; workers>1 NotImplementedError placeholder; 108 fast/router tests pass. |
| task5: `SessionRegistry.create_session_with_id` (32-hex, no overwrite) + router-side stable blake2b id gen; consistent POST /sessions routing | AC-2, AC-3 | done (pending verification) | coding | claude | M2 commit `fffa0a6ad`; new `routing.py` (blake2b, cross-process-stable, tested); needs a core-level create-with-id op in M3. |
| task6: thin router process (sole HTTP listener; extract session_id; stable-hash select; never json.loads chat body) + sticky routing (catch-all keeps no-404 proxy) | AC-2, AC-3, AC-6 | done (pending verification) | coding | claude | M3-B commit `f0a9c4808`: `session_router.py` (route order chat-before-catch-all, no json.loads, asyncio.shield, 503 backpressure); 128 tests pass, no orphans. |
| task7: IPC channel (request-id keyed, out-of-order replies, backpressure, cancellation/cleanup; status/headers/body + raw non-200 passthrough) | AC-2, AC-4, AC-5 | done (pending verification) | coding | claude | M3-A commit `606dcb1b5`: `session_ipc.py` multiplexed/single-writer/chunked; 120 tests pass incl. HOL test. |
| task8: spawn N headless workers (own asyncio loop; overlap upstream I/O; bound CPU parse via process-local semaphore) | AC-5 | done (pending verification) | coding | claude | M3-A `606dcb1b5`: `session_worker.py` (own loop, parse Semaphore, PR_SET_PDEATHSIG, ProxyBackend). |
| task9: strip R3 from client chat response; retain full R3 in worker record; verify GET-records + compute_samples_from_openai_records reconstruct R3 | AC-6 | done (pending verification) | coding | claude | M3-A `606dcb1b5`: uniform R3 strip in `session_core`; meta_info test updated; GET-records retains R3. |
| task10: non-daemon supervisor (start router+N workers, wait all-ready, /health liveness, fail-fast on worker death, graceful shutdown no orphans; preserve client-disconnect) | AC-7 | done (pending verification) | coding | claude | M3-B `f0a9c4808`: `session_supervisor.py` (spawn ctx, socketpair/worker, non-daemon, PR_SET_PDEATHSIG, fail-fast process-group kill, atexit+signal teardown); orphan check clean. |
| task11: workers=N equivalence tests + disconnect-stability tests | AC-1, AC-2, AC-3, AC-4, AC-6 | DONE (verified) | coding | claude | commits `c1485878d` (`test_session_multiprocess_equivalence.py`: 8 equivalence + 2 disconnect, end-to-end through real router+workers) + `1b338d7ad` (standalone disconnect test). Race suite stays the workers=1 proof (its in-process `patch.object` failure-injection can't cross the spawn boundary); equivalence tests cover exactly the client/backend-driven paths at N. 137 fast/router tests pass. |
| task12: port + extend overhead benchmark; OLD vs NEW p50/p95/p99 + wall time | AC-6 | DONE (verified) | coding | claude | commit `1162e8186` (HTTP-mode bench + `_mock_r3_backend.py` + 6 result JSONs) + `1b338d7ad` (SUMMARY.md). OLD w1 vs NEW w16 = 6.7× wall/throughput, p50 10.3× / p95 8.9× / p99 8.2×, ~flat RSS; w16 sweet spot. Significant win (DEC-4 directional). |
| task13: adversarial review of IPC + concurrency + R3-strip | AC-5, AC-6, AC-7 | DONE | analyze | codex | Run as a PRE-implementation Codex adversarial design review; 9 must-fixes captured as binding `m3-design-contract.md` and folded into M3. Also drove the #11 worker-death root-cause (resolved not-a-defect; harness keep-alive cascade). |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| RLCR base branch auto-detected as `main`, but this branch was cut from `9cf2a0384`; `codex review --base main` would diff against unrelated divergence. | 0 | Only affects the Review Phase comparison base, not implementation; mainline coding proceeds unaffected. | Before entering Review Phase: re-run with `--base-branch 9cf2a0384` (or set the review base to the branch point). |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| Consistent-hashing / rebalanceable routing (Alt-3) | (enhancement over AC-2) | 0 | Plan scopes v1 to simple modulo over a stable hash; consistent hashing is a pluggable upgrade for dynamic n_worker / hotspots, not first-version feasibility. | When dynamic worker scaling or hotspot long-tail is needed. |
| Delta/incremental-R3 protocol | (enhancement over AC-6) | 0 | Requires SGLang/trainer coordination outside this repo's control (DEC-4). | Separate downstream effort; a benchmark `--incremental-r3` mock already exists on the abandoned branch for reference. |
