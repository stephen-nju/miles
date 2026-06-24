Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/root/miles/.humanize/rlcr/2026-06-24_06-04-55/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/root/miles/.humanize/rlcr/2026-06-24_06-04-55/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Multi-Process Session Server: Thin Router + IPC Dispatch to Headless Workers

## Goal Description

Add an OPT-IN multi-process mode to the Miles session server so that CPU-bound response handling (chiefly `json.loads`/validation of large R3 `routed_experts` bodies) can run in parallel across OS processes and escape the GIL, instead of contending on a single process's GIL as the current bounded thread-pool does.

The architecture is a **thin router process** fronting **N headless worker processes**:

- The router is the only client-facing HTTP listener. For each request it extracts `session_id` from the path, selects `worker_idx = stable_hash(session_id) % n_worker`, forwards the request to that worker over **IPC (not a second HTTP hop)**, awaits the worker's result over IPC, and writes the HTTP response. The router holds no session state and never `json.loads` the chat response body.
- Each worker runs its own asyncio event loop and the existing session logic in-process (`SessionRegistry` / `LinearTrajectory` / per-session in-flight gate). The worker makes the upstream inference HTTP call itself (`backend_url` unchanged) and stores the full record (including R3) locally.
- Sessions are nearly standalone (no cross-session shared state); sticky-by-hash guarantees a session always returns to the same worker, so IPC moves only this-turn's request/response payload, never accumulated session state.

`--session-server-workers=1` (default) preserves today's exact single-process path (no router, no IPC), making the feature fully opt-in and backward compatible.

This plan is grounded on resolved decisions (see `## Pending User Decisions`, all RESOLVED):
- **Baseline behavior (DEC-0)**: branch from `9cf2a0384` and re-establish the per-session in-flight gate (409) and strict upstream-response validation (502) cleanly on the new branch; do NOT carry over the bounded CPU thread-pool offload (multi-process replaces it). Equivalence target = `9cf2a0384` behavior PLUS those two re-established behaviors.
- **R3 in chat response (DEC-1=B)**: strip `routed_experts` / `indexer_topk` from the client-facing chat response so the large R3 blob never crosses IPC per turn; R3 stays in the worker's stored record and is still served via `GET /sessions/{session_id}` for training-sample reconstruction.
- **Worker crash (DEC-2)**: any worker death fails the whole session server fast (loud failure; no silent shard restart that turns active sessions into wrong-state 200s).
- **v1 success (DEC-4)**: success = a correct multi-process architecture that escapes the GIL, shown as a directional improvement in aggregate response-parse throughput / tail latency on the production-shaped benchmark vs single-process and vs the thread-pool path. No fixed numeric threshold. Delta/incremental-R3 protocol is explicitly out of scope (deferred).

Branching: start from commit `9cf2a0384`; create branch `refactor/multi-process-session-server`; do NOT build on `refactor/session-server-concurrency`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Behavioral-equivalence tests should be parameterized to run against BOTH `--session-server-workers=1` (single-process adapter path) and `--session-server-workers=N` (router + IPC) deployments.

- AC-1: Opt-in and backward compatibility. With `--session-server-workers` unset or `=1`, the session server behaves byte-for-byte as the single-process server and spawns no router/extra worker processes.
  - Positive Tests (expected to PASS):
    - With workers=1, the existing public-HTTP suites (`tests/fast/router/test_sessions.py`, `tests/fast/router/test_session_race_conditions.py`, `tests/fast/router/test_session_pretokenized_e2e.py`) pass unchanged.
    - With workers=1, the launch path produces exactly one session-server process (no separate router process, no extra worker).
  - Negative Tests (expected to FAIL):
    - A configuration with workers=1 that spawns a router or any extra worker process fails the process-topology assertion.
    - Any change that alters single-process behavior for workers=1 (e.g. a different status code, dropped header) fails the unchanged suites.

- AC-2: Create-path and sticky routing consistency. With workers=N, a session created via `POST /sessions` is owned by exactly one worker, and every `session_id`-keyed endpoint routes to that same worker.
  - Positive Tests:
    - `POST /sessions` → `POST /sessions/{id}/v1/chat/completions` → `GET /sessions/{id}` → `DELETE /sessions/{id}` all operate on the same worker's state; the GET returns the record appended by the chat.
    - The catch-all `POST/GET/... /sessions/{id}/{path}` (e.g. `/abort_request`) routes to the owning worker and is proxied to the backend WITHOUT a session-existence 404 check (matching today's catch-all behavior).
  - Negative Tests:
    - A `chat/completions` / `GET` / `DELETE` for a `session_id` that no worker owns returns 404 `SessionNotFoundError` (the router must not route it to an arbitrary worker that silently creates fresh state).
    - The catch-all path returning 404 for an unknown session id (incorrectly applying the create/chat 404 rule to the proxy route) fails.

- AC-3: Stable, 32-hex session id (firm requirement). The session id remains a 32-char lowercase-hex string, and routing uses a process-stable hash independent of `PYTHONHASHSEED`.
  - Positive Tests:
    - `len(session_id) == 32` and matches `[0-9a-f]{32}` (preserves `tests/fast/router/test_sessions.py`).
    - The same `session_id` maps to the same `worker_idx` across separate processes / interpreter restarts (stable hash).
  - Negative Tests:
    - An id that is not 32-hex fails the format assertion.
    - A routing function built on Python's builtin `hash()` (per-process salted) that remaps a given id to a different worker after a process restart fails the cross-process determinism test.

- AC-4: Behavioral equivalence of the session core across transports. The decoupled core handlers produce identical session-state and error semantics whether driven via single-process FastAPI (workers=1) or via router+IPC (workers=N).
  - Positive Tests (run the race + pretokenized suites at workers=N):
    - Per-session in-flight gate: a concurrent second same-session chat returns 409 `SessionBusyError` and does NOT reach the upstream backend; different sessions run concurrently.
    - Closing (404) beats busy (409); slot is released on every error path (malformed JSON, 400 validation, transport 502, tokenization 500, invariant 500, client cancel/disconnect).
    - Non-200 upstream passes through unrecorded; invalid-200 (missing meta_info / non-numeric logprob / non-int token id) → 502 with nothing committed.
    - TITO pretokenized prefix invariants hold; rollback is at most one assistant step (`MAX_ASSISTANT_ROLLBACK_STEPS`); `append_record` prunes old R3 blobs identically; downstream framing headers (`content-length`/`transfer-encoding`/`content-encoding`) are stripped.
  - Negative Tests:
    - A worker that maps `SessionError` to the wrong status (e.g. 500 for a busy session) fails the 409 test.
    - A router/worker that records a non-200 upstream response, or commits state on an invalid-200, fails.
    - A worker that fails to release the in-flight slot on any error path fails the slot-release tests.

- AC-5: Worker concurrency model (clarifies the draft's "each worker serial"). Within a worker, different sessions' upstream inference calls overlap; CPU parse/validate is bounded; per session there is still exactly one in-flight chat.
  - Positive Tests:
    - K concurrent chats on K distinct sessions owned by the same worker, each with simulated upstream latency L, complete in ≈ L (overlapped), not ≈ K·L.
    - A `DELETE`/`GET` for one session on a worker is serviced while another session's chat on the same worker is awaiting its upstream call (the IPC dispatch is not a single FIFO that blocks other sessions behind one chat).
  - Negative Tests:
    - A strictly-serial worker that completes the K concurrent distinct-session chats in ≈ K·L fails.
    - An unbounded in-process parse concurrency that reproduces the 16-thread GIL-contention latency curve fails the bounded-parse assertion.

- AC-6: The large-response path does not relocate the bottleneck to the router (DEC-1=B). R3 is stripped from the client chat response and kept server-side; the router never parses the chat response body; router memory stays bounded under concurrency.
  - Positive Tests:
    - With R3 enabled, the client chat response contains the assistant message and small meta fields but NOT `routed_experts` / `indexer_topk`; `GET /sessions/{id}` records still contain R3 for the retained turns, and `compute_samples_from_openai_records` reconstructs samples (incl. routed_experts) unchanged.
    - The router process never calls `json.loads` on a chat response body; on the production-shaped benchmark, router RSS does not grow ~`concurrency × body_size`.
    - On the production-shaped benchmark (32 sessions, 50 turns, r3_scale=1000), workers=N shows improved aggregate response-parse throughput / tail latency vs workers=1 and vs the thread-pool path (directional; per DEC-4).
  - Negative Tests:
    - A client chat response that still includes the full `routed_experts` blob fails the R3-strip assertion.
    - A router that `json.loads`/re-serializes the full response dict, or whose RSS scales linearly with `concurrency × R3_size` to OOM risk, fails.
    - A change that breaks `GET`-records R3 retrieval (so training-sample reconstruction loses routed_experts) fails.

- AC-7: Lifecycle, health, and crash semantics. The launcher starts a router + N workers under a non-daemon supervisor; readiness waits for every worker's tokenizer/TITO init; the router `/health` reflects worker liveness; shutdown leaves no orphan processes; any worker death fails the whole session server fast (DEC-2).
  - Positive Tests:
    - Startup blocks until all workers report ready; `OpenAIEndpointTracer.create()`'s `/health` (with the stable `session_server_instance_id` shape) succeeds against the router.
    - `SIGTERM` to the supervisor terminates the router and all workers with no orphaned processes.
    - Killing one worker causes the session server to fail fast (router stops serving / surfaces unhealthy), rather than silently restarting an empty worker that answers later requests with fresh state.
    - A real client disconnect mid-chat does not leak the worker's in-flight slot or a router-side pending reply (matching today's "handler continues and may commit the turn" semantics, with the slot released on completion).
  - Negative Tests:
    - Orphaned worker processes remaining after supervisor shutdown fails.
    - A dead shard being silently replaced by a fresh empty worker that returns 200 for a previously-existing session fails the fail-fast policy.
    - A leaked in-flight slot or pending IPC reply future after client disconnect fails.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A complete opt-in multi-process session server: a thin router process (sole HTTP listener) dispatching by a stable hash of `session_id` over IPC to N headless asyncio workers; router-owned 32-hex session ids with consistent create/chat/get/delete/catch-all routing; session handlers decoupled from FastAPI into transport-neutral core callables; per-worker overlapped upstream I/O with bounded CPU parse; re-established per-session in-flight gate (409) and strict upstream-response validation (502); R3 stripped from the client chat response and retained only in worker records (still served via GET records); a non-daemon supervisor with all-workers-ready startup, `/health` liveness, fail-fast-on-worker-death, and graceful shutdown with no orphans; and a production-shaped benchmark demonstrating the parallel-parse win across single-process / thread-pool / multi-process. Behavioral-equivalence suites pass at workers=1 and workers=N.

### Lower Bound (Minimum Acceptable Scope)
The same opt-in multi-process server using simple `% n_worker` over a stable hash (no consistent hashing), router-owned 32-hex ids, decoupled core handlers, sticky routing for all session-keyed endpoints, the re-established in-flight gate + strict validation, R3 stripped from the client response, workers=1 unchanged, AC-1..AC-5 and AC-7 satisfied, and at least a correctness benchmark plus a directional perf measurement for AC-6. The IPC transport may be the simplest mechanism that satisfies AC-6 (router never parses the chat body; bounded router RSS; worker concurrency not serialized).

### Allowed Choices
- Can use: a process-stable hash via `hashlib` (e.g. `blake2b`/`sha256`) for routing; IPC via UNIX-domain socket / `socketpair` framed binary protocol, OR `multiprocessing` `Pipe`/`Queue`, OR shared memory — provided AC-5 and AC-6 hold (no worker serialization, router never `json.loads` the chat body, bounded router RSS); `multiprocessing` spawn context for child processes; a thin FastAPI adapter for the workers=1 path that calls the same decoupled core; a retained `SessionServer.do_proxy` facade to minimize test monkeypatch churn.
- Cannot use: Python builtin `hash()` as the routing protocol (per-process salted); a second loopback HTTP hop on the primary's hot path; reuse of `MilesRouter`'s `httpx` + `json.loads` + `JSONResponse` proxy core for the chat path; sharding session state without sticky single-owner per session; a daemon router/worker that itself spawns children (daemonic processes cannot have children); a single FIFO IPC dispatch per worker that blocks other sessions' requests behind one in-flight chat; any change that breaks `GET`-records R3 retrieval used by training-sample reconstruction.

> Note: The HTTP-reverse-proxy fallback (each worker a full unmodified `session_server`, router proxying by hash) is explicitly OUT of scope here because it contradicts the firm "no second HTTP hop" position. If decoupling the FastAPI handlers proves infeasible, stop and re-plan rather than silently switching to that fallback.

## Feasibility Hints and Suggestions

> Note: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
One possible decomposition:

```
[ rollout client / agent ]
        │  HTTP  (POST /sessions, /sessions/{id}/v1/chat/completions, GET, DELETE, /{path})
        ▼
[ thin router process ]  ── sole HTTP listener; no session state; never json.loads chat body
        │  extract session_id from path; worker_idx = blake2b(session_id) % n_worker
        │  IPC: send (request_id, method, path, query, headers, body) ; await reply
        ▼
[ worker_i process ]  ── own asyncio loop; in-process SessionRegistry/LinearTrajectory/in-flight gate
        │  core handler (transport-neutral): claim per-session slot → TITO prepare under lock
        │  → upstream HTTP call to backend_url (overlaps across sessions) → validate → commit record (full R3)
        │  → reply over IPC: status, headers, body  (R3 STRIPPED from this client-facing body)
        ▼
   GET /sessions/{id}  → returns records WITH R3 (once per session, at collection time)
```

- `POST /sessions`: the ROUTER generates the 32-hex id, computes the owner, and sends an explicit "create with this id" to that worker (`SessionRegistry.create_session_with_id(session_id)`), so the create routes consistently with later requests.
- Decouple `setup_session_routes` closures into transport-neutral core functions taking `(method, path, query, headers, body)` and returning a typed result `(status_code, headers, body_bytes, raw_passthrough_flag)`; the worker maps `SessionError`→status; non-200 passthrough stays unrecorded; the router only writes the HTTP response.
- Worker concurrency: keep the per-session in-flight gate; let different sessions' upstream `await`s overlap on the worker's loop; bound CPU parse/validate with a process-local semaphore (size ~1–2), NOT a thread pool that re-creates GIL contention.
- The IPC dispatch must support out-of-order replies keyed by `request_id` so a `DELETE`/`GET` is not blocked behind an in-flight chat on the same worker.
- workers=1 short-circuits: keep a thin FastAPI adapter wrapping the same core so existing HTTP tests run unchanged and no router/IPC is created.

### Relevant References
- `miles/rollout/session/session_server.py` — `SessionServer`, `do_proxy`, `build_proxy_response`, `run_session_server`; the "single uvicorn worker on purpose / sticky ownership deferred" comment is the motivation.
- `miles/rollout/session/sessions.py` — `setup_session_routes` (the FastAPI closures to decouple), `chat_completions` (in-flight gate, lock discipline, non-200 passthrough, validation offload), `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response` (already pure functions).
- `miles/rollout/session/linear_trajectory.py` — `SessionRegistry` (`create_session`; needs `create_session_with_id`), `LinearTrajectory` (lock, `chat_inflight`, `prepare_pretokenized`, `update_pretokenized_state`, `append_record` R3 pruning, rollback).
- `miles/rollout/session/session_errors.py` / `session_types.py` — `SessionError`→status mapping; Pydantic `SessionRecord`.
- `miles/ray/rollout/router_manager.py` — `start_session_server` (single launch point; today `multiprocessing.Process(..., daemon=True)`), `start_router`; `miles/ray/rollout/rollout_manager.py` calls `start_session_server`.
- `miles/rollout/generate_utils/openai_endpoint_utils.py` — `OpenAIEndpointTracer` (`create()` POSTs `/sessions`, builds `base_url`, GETs records, DELETEs); `compute_samples_from_openai_records` reads R3 from stored records via `get_routed_experts_from_response`.
- `miles/utils/arguments.py` — `add_session_arguments` (~`--session-server-*` block) for adding `--session-server-workers`; `--p2p-transfer-num-workers` (default 4) as a worker-count naming precedent.
- `miles/utils/http_utils.py` — `find_available_port`, `is_port_available`, `wait_for_server_ready` (reuse for lifecycle).
- `tests/fast/router/test_sessions.py`, `test_session_race_conditions.py`, `test_session_pretokenized_e2e.py` — public-HTTP behavioral suites to parameterize across workers.
- `tests/benchmark/bench_session_server_overhead.py`, `bench_session_responsiveness.py`, `docs/developer/session-server-overhead.md` — benchmark harness + measured evidence (parse-dominated cost, GIL).
- `miles/router/router.py` — `MilesRouter`; reference its bookkeeping (worker registry / health / dead-worker isolation) ONLY; do NOT reuse its `json.loads` proxy core.

## Dependencies and Sequence

### Milestones
1. Milestone M0 — Baseline + behavior re-establishment: Branch `refactor/multi-process-session-server` from `9cf2a0384`. Re-establish, cleanly, the per-session in-flight gate (409) and strict upstream-response validation (502/UpstreamResponseError) on this branch; do NOT add the bounded CPU thread-pool offload. This fixes the single-process equivalence target that AC-4 references.
   - Phase A: Cut the branch from `9cf2a0384`; confirm the baseline session behavior (no gate / no strict validation / raw passthrough present).
   - Phase B: Re-add per-session single in-flight semantics + strict validation (no thread pool); update/port the corresponding tests as the equivalence baseline.
2. Milestone M1 — Decouple session core from FastAPI: Extract transport-neutral core handlers (create/get/delete/chat/proxy) taking `(method, path, query, headers, body)` → typed result, preserving query forwarding and request/response header stripping. Keep a thin FastAPI adapter so workers=1 (and all existing HTTP tests) work unchanged.
   - Phase A: Introduce the core handler module + typed result; map `SessionError`→status outside FastAPI.
   - Phase B: Re-wire `setup_session_routes` / `SessionServer` to call the core via the adapter; preserve `do_proxy` facade.
3. Milestone M2 — Worker-count arg + router-owned stable id + create routing: Add `--session-server-workers` (default 1 → single-process path). Add `SessionRegistry.create_session_with_id` (validate 32-hex, no overwrite, collision-safe) and router-side stable-hash id generation so `POST /sessions` routes consistently.
4. Milestone M3 — Router + IPC + headless workers + sticky routing: Build the thin router process and the IPC channel (out-of-order, request-id keyed, backpressure, cancellation/cleanup). Spawn N headless workers (no HTTP listener). Sticky-route all session-keyed endpoints; carry `SessionError`/status and raw non-200 passthrough across IPC without router-side parsing. Strip R3 from the client chat response (DEC-1=B), keep full R3 in the worker record.
   - Depends on M1 (core handlers) and M2 (arg + id + create routing).
5. Milestone M4 — Concurrency, lifecycle, crash policy: Per-worker overlapped upstream I/O + bounded parse (AC-5). Non-daemon supervisor topology (router + N workers), all-workers-ready startup, `/health` liveness aggregation, graceful shutdown without orphans, and fail-fast-on-worker-death (DEC-2). Preserve client-disconnect semantics without slot/reply leaks.
   - Depends on M3.
6. Milestone M5 — Equivalence + benchmark + perf validation: Parameterize the public-HTTP suites across workers=1/N (AC-4). Extend the benchmark to compare `9cf2a0384` single-process, the thread-pool path, and multi-process; validate the directional parallel-parse win and bounded router RSS (AC-6, DEC-4).
   - Depends on M3 and M4.

Dependency summary: M0 → M1 → M2 → M3 → {M4, M5}; M5 also depends on M4 for the lifecycle/concurrency behaviors it benchmarks.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude; `analyze` = executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Cut `refactor/multi-process-session-server` from `9cf2a0384`; document the baseline session behavior delta vs current HEAD | AC-1, AC-4 | coding | - |
| task2 | Re-establish per-session in-flight gate (409) + strict upstream-response validation (502) on the new branch, WITHOUT the bounded CPU thread pool | AC-4 | coding | task1 |
| task3 | Extract transport-neutral core handlers (create/get/delete/chat/proxy) with typed result + `SessionError`→status mapping; preserve query/header stripping; keep a thin FastAPI adapter + `do_proxy` facade for workers=1 | AC-1, AC-4 | coding | task2 |
| task4 | Add `--session-server-workers` (default 1) in `add_session_arguments`; wire through `router_manager`/`run_session_server`; workers=1 short-circuits to single-process path | AC-1 | coding | task3 |
| task5 | Add `SessionRegistry.create_session_with_id` (validate 32-hex, no overwrite) + router-side stable-hash (`blake2b`) id generation; make `POST /sessions` route consistently | AC-2, AC-3 | coding | task4 |
| task6 | Implement the thin router process (sole HTTP listener; extract session_id; stable-hash select; never `json.loads` chat body) + sticky routing for chat/get/delete/catch-all (catch-all keeps no-404 proxy semantics) | AC-2, AC-3, AC-6 | coding | task5 |
| task7 | Implement the IPC channel: request-id keyed, out-of-order replies, backpressure, cancellation/cleanup; carry status/headers/body + raw non-200 passthrough across IPC | AC-2, AC-4, AC-5 | coding | task6 |
| task8 | Spawn N headless workers (no HTTP listener) running the core via their own asyncio loop; overlap different-session upstream I/O; bound CPU parse with a process-local semaphore | AC-5 | coding | task7 |
| task9 | Strip R3 (`routed_experts`/`indexer_topk`) from the client-facing chat response; retain full R3 in the worker record; verify GET-records + `compute_samples_from_openai_records` still reconstruct R3 | AC-6 | coding | task8 |
| task10 | Non-daemon supervisor topology: start router + N workers, wait all-ready, `/health` liveness, fail-fast on worker death, graceful shutdown without orphans; preserve client-disconnect (no slot/reply leak) | AC-7 | coding | task8 |
| task11 | Parameterize `tests/fast/router/test_sessions.py`, `test_session_race_conditions.py`, `test_session_pretokenized_e2e.py` to run at workers=1 and workers=N; add create-path/sticky-routing, stable-hash-determinism, and R3-strip tests | AC-1, AC-2, AC-3, AC-4, AC-6 | coding | task9, task10 |
| task12 | Extend the benchmark harness to compare `9cf2a0384` single-process, thread-pool path, and multi-process; report throughput/parse p50-p95/router RSS/worker RSS; validate directional win + bounded router RSS | AC-6 | coding | task10 |
| task13 | Adversarial review of the IPC transport + concurrency design (deadlock/backpressure/cancellation/large-body RSS) and the R3-strip behavior-change blast radius across all session consumers | AC-5, AC-6, AC-7 | analyze | task9, task10 |

## Claude-Codex Deliberation

### Agreements
- Multi-process needs sticky single-owner routing per session because `SessionRegistry` is an in-process dict and `LinearTrajectory.lock`/`chat_inflight` are in-process state (the code's own "sticky ownership deferred" comment confirms it).
- Router-owned, stable, 32-hex session id; workers must support "create with id".
- The worker must make the upstream inference HTTP call itself; the router must never `json.loads` the chat response body (otherwise the bottleneck just relocates to the router).
- workers=1 must short-circuit to today's single-process path; opt-in with zero behavioral risk.
- Decoupling the FastAPI closures into transport-neutral core handlers is the correct first structural step; simple modulo (over a stable hash) is acceptable for v1, with consistent hashing deferred.
- The worker must not be strictly serial across sessions — serialize only per-session mutation and bound CPU parse; let different sessions' upstream I/O overlap.

### Resolved Disagreements
- Baseline contents (high impact): Codex flagged — and git confirms — that `9cf2a0384` lacks the in-flight gate / strict validation / CPU executor (they live above it on the abandoned branch). Resolution: added M0 to re-establish the gate + strict validation cleanly on the new branch (drop the thread pool), per user decision DEC-0. AC-4's equivalence target is this re-established behavior set, not current HEAD.
- "Behavior identical" vs R3 stripping: reconciled — equivalence covers session state, error/gate/validation semantics, and training-sample reconstruction; the client-facing chat response intentionally differs (R3 stripped, DEC-1=B) and is the single documented behavior change, with one test assertion updated.
- Catch-all 404: corrected — `/sessions/{id}/{path}` must NOT 404 on unknown id; it sticky-routes by session_id and proxies `{path}` (preserving `/abort_request` behavior). AC-2 negative test added.
- Process topology: corrected — today's launch is `daemon=True` (fork); a daemon cannot spawn children, so the design uses a non-daemon supervisor owning router + N worker PIDs with explicit startup/SIGTERM/join/no-orphan handling (AC-7).
- Pre-body 409 fast-fail: relaxed to the real contract — a concurrent same-session chat returns 409 and does NOT enter the backend; the "reject before reading the body" micro-optimization is an allowed implementation detail, not a hard requirement.
- HTTP-reverse-proxy fallback (Alt-1): removed from in-scope allowed choices (conflicts with the firm "no second HTTP hop"); if handler decoupling proves infeasible, stop and re-plan.

### Convergence Status
- Final Status: `converged`
- Rounds: Codex first-pass analysis (1) + one convergence review round (1). All of Codex's REQUIRED_CHANGES were accepted and folded in; the only remaining opposite-opinion items were user decisions, now all resolved (DEC-0..DEC-5). No contested Claude/Codex disagreement remains.

## Pending User Decisions

- DEC-0: Baseline behavior scope for the new branch (gate / validation / thread pool).
  - Claude Position: Re-establish the per-session in-flight gate + strict response validation; drop the thread-pool offload (multi-process replaces it).
  - Codex Position: Required a baseline-clarifying M0; either change base or treat the gate/passthrough/validation as M0 prerequisites since `9cf2a0384` lacks them.
  - Tradeoff Summary: Re-establishing keeps correctness behaviors the team valued while abandoning the GIL-bound thread-pool effort; pure-`9cf2a0384` would be simpler but drop the 409 fast-fail and 502 hardening.
  - Decision Status: RESOLVED — Re-establish gate + validation, drop the thread pool.

- DEC-1: Whether the client-facing chat response keeps full R3 or strips it.
  - Claude Position: Strip R3 from the client response, keep it in the worker record (served via GET records). Realizes the thin-router/IPC win; the agent never reads R3 from the chat response.
  - Codex Position: A reasonable product change but not "behavior identical"; if chosen, reframe ACs to training-sample equivalence + intentional client-response R3 removal and verify GET-records still serves R3.
  - Tradeoff Summary: Strip = biggest hot-path win (blob never crosses IPC per turn), costs one updated passthrough test; keep-full = no test change but the 100+ MiB blob crosses IPC each turn and pressures router memory.
  - Decision Status: RESOLVED — Strip from the client chat response (DEC-1=B); R3 retained in worker records and served via GET.

- DEC-2: Worker crash semantics.
  - Claude Position: Fail-fast or shard-503; never silently restart into an empty worker that returns wrong-state 200s.
  - Codex Position: Must be decided before M4 since `/health`, error mapping, orphan cleanup, and active-session-loss semantics depend on it.
  - Tradeoff Summary: Fail-fast is the loudest, simplest invariant; shard-503 keeps other shards alive; restart-and-lose risks silent breakage.
  - Decision Status: RESOLVED — Fail-fast the whole session server on any worker death.

- DEC-3: Session id appearance.
  - Claude Position: Keep 32-hex (router-generated, stable-hashed).
  - Codex Position: Make it a firm requirement, not an open decision — tests and client metadata assume the 32-hex shape.
  - Tradeoff Summary: 32-hex preserves the existing test/consumer contract; embedding a worker index would change the shape and break assertions.
  - Decision Status: RESOLVED (firm) — Keep 32-hex; stable hash for routing.

- DEC-4: v1 success criterion / perf gate.
  - Claude Position: v1 = correct multi-process architecture with a directional parallel-parse improvement on the production-shaped benchmark; delta-R3 protocol deferred.
  - Codex Position: The success goal must be fixed because it changes the acceptance benchmark; benchmark must compare `9cf2a0384`, thread-pool, and multi-process.
  - Tradeoff Summary: Directional keeps v1 shippable and avoids cross-repo protocol coupling; a hard number or mandatory R3-protocol change would expand scope and external dependencies.
  - Decision Status: RESOLVED — Directional architecture + parallel-parse win; no fixed numeric threshold; delta-R3 deferred.

- DEC-5: IPC transport mechanism.
  - Claude Position: Treat as an allowed implementation choice gated by AC-6 (router never `json.loads` the chat body; bounded router RSS) and AC-5 (no worker serialization).
  - Codex Position: Implementer may choose, but acceptance must fix: router does not parse the chat body, RSS bounded, worker concurrency not degraded; prefer UDS/socketpair framing over a pickling Queue for any large body.
  - Tradeoff Summary: With DEC-1=B the large blob no longer crosses IPC per turn, so transport risk drops sharply; the AC gates keep any choice honest.
  - Decision Status: RESOLVED — Allowed choice gated by AC-5/AC-6; with DEC-1=B, transport is low-risk for the per-turn hot path.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `session_server_workers`, `worker_index`, `route_session`, `create_session_with_id`).

### Behavior-change note
- The single intended client-visible behavior change is the removal of `routed_experts`/`indexer_topk` from the client-facing chat completion response (DEC-1=B). All other client-visible behavior (status codes, error bodies, header handling, session lifecycle, GET-records contents including R3) must remain equivalent to the M0 single-process baseline.

--- Original Design Draft Start ---

# Session Server 多进程化：薄 Router 经 IPC 分发到独立 Worker

## Original Idea

当前分支做了很多 session server 里面做多线程/多进程的努力。但是是徒劳的，没用，一直卡在 gil，没什么优化空间。每轮只返回增量 r3 需要 further discussion，目前做不了。现在有个 idea 是，可以做整个 session server 级别的多进程，举个例子：
1. 我们吧 session server 拆分成 router 和 worker。我们预先启动多个 worker 进程，比如 n_worker。每当有新的 session id 进来时，就走 router，把请求路由到这个 session id 取模 n_worker 上面，用它来实际处理和转发请求。然后每个 worker 内部，依然是串行的。
这么做成立的主要原因是，session id 之间是几乎 standalone 的，所以不用担心内存在进程间 transfer 开销的问题。
2. 你从 9cf2a0384 为基准开始做。不要再 refactor/session-server-concurrency 上面改。开个新分支叫 refactor/multi-process-session-server

## Primary Direction: 薄 Router + 进程间 IPC 分发（router 独占网络）

### Rationale

router 极薄——只做"终止客户端 HTTP + 按 session_id 取模选 worker + 进程间转发"，worker 无网络监听、是纯处理进程；router→worker 走 IPC 而非第二跳 HTTP，这与"HTTP 反代"形态是根本区别，也避开了在 router 处对 R3-heavy 响应再拷贝/再解析的浪费。

### Approach Summary

把当前单进程 session server 拆成"一个薄 router + N 个无网络 worker"，唯一新增的跨进程跳是 IPC，不是 loopback HTTP：

- **Router 进程（薄，async）**：唯一面向 rollout 客户端的 HTTP 监听者。收到 `/sessions/{session_id}/...` 后只做四件事——读取 body、从 path 提取 `session_id`、`worker_idx = hash(session_id) % n_worker`、把 `(method, path, body, headers)` 经 IPC 通道发给 worker_idx 并 `await` 其结果，最后回写 HTTP 响应。router 不持有任何 session 状态、不解析响应体、不再发起第二跳 loopback HTTP。
- **N 个 Worker 进程（无网络监听，各跑自己的 asyncio loop）**：经 IPC 收到被分发的请求 → 执行现有 session 处理逻辑（`SessionRegistry` / `LinearTrajectory` / per-session in-flight gate 全部保留在 worker 进程内）→ 其中对**上游推理引擎**（sglang / miles 推理 router）的调用仍是真实 HTTP（`SessionServer.do_proxy` 的 `backend_url`，保持不变）→ 把响应经 IPC 回传给 router。
- **为何 IPC 而非 HTTP 是根本区别（两个 router 目标不同，不可混为一谈）**：miles 推理 router 的目标是把请求分发到分布在多 GPU / 多节点上的 SGLang 引擎，target 天然可能跨机，所以**必须**走 HTTP——HTTP 对它是正确选择；而 session-server router 的目标完全不同，是把请求分发到**同机的 sibling worker 进程**（session 之间 standalone、都在一台 host 上），因此应走 IPC。在 HTTP 反代形态里，含 R3 blob 的大响应还要在 router 处被完整缓冲、甚至 `json.loads` 再重建一遍（见 Objective Evidence 对 `router.py` 的实证）；改成 IPC 后，router 退化为字节搬运 + 哈希选路，热路径上不再有多余的 JSON 解析与 HTTP 往返。
- **与原 idea 论证一致**：session 之间 standalone，sticky-by-hash 保证同一 session 永远回到同一 worker，所以 IPC 只搬"这一次请求/响应"的 payload，不搬 session 累积状态——不存在进程间 transfer 累积状态的开销。
- 改动落点：新建薄 router；把 session handler 从 FastAPI 路由解耦为可由 IPC 消息直接驱动的纯 async 函数（见 Known Risks）；`miles/ray/rollout/router_manager.py` 的 `start_session_server` 改为 spawn 1 个 router + N 个 headless worker 并建立 IPC 通道；新增类似 `--n-session-server-workers` 的参数。

### Objective Evidence

- **反证"为何不用 HTTP 反代 / 不复用 MilesRouter"**：`miles/router/router.py:130-158` 的 `MilesRouter.do_proxy()` 用 `httpx` 向 worker URL 发起完整 HTTP 请求并 `await response.aread()` 缓冲整个响应体；`build_proxy_response()`（L160-171）进一步对响应体做 `json.loads(content)` 再重建 `JSONResponse`——即把含 R3 blob 的大响应在 router 处完整再解析一遍。注意 MilesRouter 之所以是 HTTP，是因为它的**目标不同**：它要把推理请求分发到可能跨 GPU/跨节点的 SGLang 引擎，跨机就必须 HTTP；而 session-server router 的目标是同机 sibling 进程分发，目标不同 → 传输选择也不同。所以 MilesRouter 的 HTTP 设计对它自己正确，但不适合照搬做本 router 的底座。
- `miles/rollout/session/session_server.py:50-86` 的 `do_proxy` 把请求转发到 `backend_url`（推理引擎），不是转发到 worker；多进程下 worker 承载的正是这套 session + do_proxy 逻辑，对上游引擎的 HTTP 调用保持不变。
- `miles/rollout/session/session_server.py:117-120` 注释明确写道 "Single uvicorn worker on purpose… Multi-process needs sticky session ownership and is deferred"——薄 router + hash 分发正是为补上这条 "sticky session ownership"。
- `miles/ray/rollout/router_manager.py` 已用 `multiprocessing.get_context("spawn").Process` 启动 session server 进程；循环 spawn N 个 headless worker 并在父进程建立 IPC 通道是其自然延伸（spawn 已避免 fork 继承线程的问题）。
- `miles/rollout/session/sessions.py:40-109` 的 `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response` 已是纯函数并在 `backend.cpu_executor` 中离线执行，便于从 FastAPI 请求对象解耦、改由 IPC 消息驱动。
- `miles/rollout/session/session_types.py` 的 `SessionRecord` 是 Pydantic BaseModel、请求/响应均为 dict，payload 跨 IPC 可序列化（请求侧每 turn 约数百字节；响应侧大小取决于 R3，见 Known Risks）。
- `miles/rollout/session/linear_trajectory.py:248-302` —— `SessionRegistry` 是纯进程内 `dict[session_id -> LinearTrajectory]`，session 间无共享状态，天然支持落在各自 worker 进程内。

### Known Risks

- **session handler 与 FastAPI 解耦（IPC 形态的主要内部成本）**：现有处理逻辑挂在 `setup_session_routes(self.app, ...)` 的 FastAPI 路由上。要由 IPC 消息驱动，需把 handler 从 `fastapi.Request` / `starlette.Response` 解耦成可直接以 `(method, path, body, headers)` 调用的纯 async 函数（或在 worker 内跑一个极小的内部 dispatch）。这是薄 router/IPC 形态相对"HTTP 反代"形态多付的改造成本，也是与保底退路（Alt-1）的主要权衡点。
- **IPC 机制选择 = 性能分水岭（需 further discussion，plan 阶段定型）**：响应体可能很大（R3 blob ~1KB/token，`docs/developer/session-server-overhead.md` 实测保留量可达多 GB）。用 `multiprocessing.Queue` / `Pipe` 会对大响应做 pickle/拷贝，可能抵消掉省下的 HTTP 一跳；用共享内存（`multiprocessing.shared_memory` / mmap）或 UNIX domain socket 传字节流更省。这条直接决定方案是否真优于 HTTP 反代——且直接呼应原 idea 的"每轮只返回增量 r3"：若能只回传增量 R3，IPC payload 会显著变小，该方向收益更明确。
- **worker 内并发语义**：原 idea 写"每个 worker 内部依然串行"。若严格逐请求串行，一个 shard 内多个 session 会被一次上游推理调用阻塞、吞吐受限；更可行的是 worker 跑自己的 asyncio loop 让不同 session 的上游 I/O 重叠（in-flight gate 仍 per-session），而 CPU 工作单线程、跨 worker 才并行（这才是真正绕开 GIL 的点）。需在 plan 阶段把"串行"的确切粒度定清楚。
- **背压与队列容量**：IPC 通道需设上限并处理满/超时；per-session in-flight gate 可缓解同一 session 的堆积，但同一 worker 的多个 session 仍可能堆积。
- **生命周期与可观测性**：headless worker 不再有 HTTP `/health`，需改用进程存活/心跳做健康检查；日志按 worker 分层；统一 graceful shutdown 避免孤儿进程。
- **路由一致性 / 资源倍增**：router 与 worker 对 `session_id` 的哈希口径须一致，否则路由到不拥有该 session 的 worker 会返回 404（`SessionNotFoundError`）；每个 worker 各自加载 tokenizer、各自持 `SessionRegistry`，内存随 `n_worker` 线性增长，`session_server_cpu_workers` 语义变为 per-worker。

## Alternative Directions Considered

### Alt-1: HTTP 反代 + 独立 worker 服务（含复用 MilesRouter，作保底退路）
- Gist: 每个 worker 就是一份完整、未改动的 `session_server` 实例，各监听 `127.0.0.1` 独立端口；router 用 `httpx` 按 `hash(session_id) % n_worker` 反向代理转发。可直接拿 `MilesRouter` 当底座，仅把 `_use_url()` 的最小负载选择换成哈希取模。最大优点是 `sessions.py` / `linear_trajectory.py` 内核零改动、最快跑通。（原稿中"复用 MilesRouter"已并入此项，因二者本质同为 HTTP 反代，避免重复。）
- Objective Evidence:
  - `miles/router/router.py:16-228` 的 `MilesRouter` 已具备 worker 注册、健康检查循环、死 worker 隔离与 `/{path:path}` catch-all 反代，改选择函数即可。
  - `tests/e2e/sglang/test_r3_router_equivalence.py` 与 `tests/fast/router/test_router.py` 已对该 router 的 worker 管理做生产级/单元测试。
- Why not primary: 用户明确指出 router 应极薄、走进程间转发而非 HTTP 转发。该形态多一跳 loopback HTTP，且 router 会把含 R3 blob 的大响应缓冲并 `json.loads` 再重建（`router.py:160-171` 实证），热路径开销大。仅作"若 handler 解耦成本过高"时的保底退路与可行性验证起点。

### Alt-2: SO_REUSEPORT / fd 传递预 fork 进程池
- Gist: 预 fork `n_worker` 个进程共享同一监听 socket（`SO_REUSEPORT` 或 fd 继承），靠 OS 级负载均衡省掉 router 转发，再在其上叠加 `session_id` 亲和（router 接受连接后传 fd，或 worker 自行按哈希取舍）。
- Objective Evidence:
  - `miles/utils/http_utils.py:18-40` 已有 `socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)` 的 socket 操作模式；`uvicorn.run()` 支持 `fd` 参数，但仓库现以黑盒 `uvicorn.run()` 启动，未启用。
- Why not primary: OS LB 默认不保证 session 亲和，必须额外补一层亲和路由；改动深入 uvicorn/socket 启动方式、可移植性差，性价比低（探索自评 low）。与 primary 相比，它连"请求级转发"都省了（直接连接级），但亲和与生命周期更难控。

### Alt-3: 一致性哈希可重平衡路由
- Gist: 把路由/亲和策略做成一等组件——带虚拟节点的一致性哈希环替代简单取模，支持 `n_worker` 动态增减时仅重映射约 `1/n_worker` 的 session、热点感知与 graceful drain，与传输层（IPC/HTTP）正交。
- Objective Evidence:
  - `miles/backends/sglang_utils/arguments.py:22-26` 已支持 `--sglang-router-policy consistent_hashing`；`miles/rollout/sglang_rollout.py:195-198` 用 `X-SMG-Routing-Key` header 传 `session_id`，证明本生态已落地一致性哈希。
  - 当前 `miles/router/router.py` 仅有最小负载选择；`miles/utils/arguments.py` 尚无 `n_session_server_workers` 类参数。
- Why not primary: 它解决的是"扩缩容/热点"进阶问题而非首版可行性，更适合作为 primary 路由策略的可插拔升级（首版用简单取模）。

### Alt-4: 无状态 worker + 外置 session 存储
- Gist: 反转 pin 前提——把 session 状态外置到共享存储（shm / Redis / 文件），worker 无状态、任意 worker 服务任意 session，router 自由均衡且天然容错；每轮 load → 处理 → save。
- Objective Evidence:
  - `docs/developer/session-server-overhead.md` 量化：主导开销是 response body JSON 解析（16-worker 下 parse p50 ≈ 3.36s），而非状态共享；外置存储的 serialize/deserialize 会**放大**该开销。
  - 同文档：未裁剪时 R3 blob 可达约 78 GB（裁剪后约 6 GB），说明 session state 体量大、跨进程搬运成本高；仓库内无分布式 session 存储 / 跨进程 CAS 的现成实现。
- Why not primary: 直接否定了"session standalone 所以 pin 便宜"的收益前提——既不解决 GIL，又新增序列化/分布式锁成本，还要从零设计一致性协议，风险与工作量最高（探索自评 low）。主要价值是前置告警（见 Synthesis Notes）。

## Synthesis Notes

primary 选定后，几条仍值得带进 plan 阶段的取舍：(1) **Alt-1 是 primary 的保底退路**——若把 session handler 从 FastAPI 解耦的成本过高，可退回"每 worker 跑整份现有 `session_server` + router 用 `httpx` 反代（甚至直接复用 `MilesRouter` 改选择函数）"，代价是多一跳 HTTP + 大响应再 parse，但内核零改动、能最快跑通；建议先用它做可行性验证，再切到 IPC 省热路径。(2) **MilesRouter 仅剩 bookkeeping 思路可借**——其 worker 注册表/健康检查/死 worker 隔离的结构可参考，但 proxy 核心（`httpx` + `json.loads` 重建）对 primary 不适用、不应复用；primary 的 router 是全新的薄进程。(3) **IPC 机制是 primary 成败关键**——应在 plan 阶段先用 `docs/developer/session-server-overhead.md` 的基准量化"大响应跨 IPC 的拷贝/序列化成本"，再在 `Queue`/`Pipe` vs 共享内存 vs UDS 间定型；这也与原 idea 的"每轮只返回增量 r3 需要 further discussion"强相关——若能只回传增量 R3，IPC payload 大幅缩小，primary 收益最明确，因此"增量 R3"值得作为并行子课题推进。(4) **Alt-3 一致性哈希**作为 primary 路由策略的可插拔升级，首版简单取模、出现扩缩容/热点长尾时再启用。(5) **Alt-4** 作为前置告警：投入多进程前先用基准确认瓶颈确实是 GIL/串行，而非 response JSON 解析本身，否则任何进程级方案收益都会被高估。

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/root/miles/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/root/miles/.humanize/bitlesson.md

## Agent Teams Mode

You are operating in **Agent Teams mode** as the **Team Leader** within an RLCR (Review-Loop-Correct-Repeat) development cycle.

This is the initial round. Read the implementation plan thoroughly before creating your team. Key RLCR files to be aware of:
- **Plan file** (provided above): The full scope of work and requirements your team must implement
- **Goal tracker** (`goal-tracker.md`): Tracks acceptance criteria, task status, and plan evolution - read it before splitting tasks
- **Work summary**: After all teammates finish, you must write a summary of what was accomplished into the designated summary file

### Your Role

You are the team leader. Your ONLY job is coordination and delegation. You must NEVER write code, edit files, or implement anything yourself.

Your primary responsibilities are:
- **Split tasks** into independent, parallelizable units of work
- **Create agent teams** to execute these tasks using the Task tool with `team_name` parameter
- **Coordinate** team members to prevent overlapping or conflicting changes
- **Monitor progress** and resolve blocking issues between team members
- **Wait for teammates** to finish their work before proceeding - do not implement tasks yourself while waiting

If you feel the urge to implement something directly, STOP and delegate it to a team member instead.

### Guidelines

1. **Task Splitting**: Break work into independent tasks that can be worked on in parallel without file conflicts. Each task should have clear scope and acceptance criteria. Aim for 5-6 tasks per teammate to keep everyone productive and allow reassignment if someone gets stuck.
2. **Cold Start**: Every team member starts with zero prior context (they do NOT inherit your conversation history). However, they DO automatically load project-level CLAUDE.md files and MCP servers. When spawning members, focus on providing: the implementation plan or relevant goals, specific file paths they need to work on, what has been done so far, and what exactly needs to be accomplished. Do not repeat what CLAUDE.md already covers.
3. **File Conflict Prevention**: Two teammates editing the same file causes silent overwrites, not merge conflicts - one teammate's work will be completely lost. Assign strict file ownership boundaries. If two tasks must touch the same file, sequence them with task dependencies (blockedBy) so they never run in parallel.
4. **Coordination**: Track team member progress via TaskList and resolve any discovered dependencies. If a member is blocked or stuck, help unblock them or reassign the work to another member.
5. **Quality**: Review team member output before considering tasks complete. Verify that changes are correct, do not conflict with other members' work, and meet the acceptance criteria.
6. **Commits**: Each team member should commit their own changes. You coordinate the overall commit strategy and ensure all commits are properly sequenced.
7. **Plan Approval**: For high-risk or architecturally significant tasks, consider requiring teammates to plan before implementing (using plan mode). Review and approve their plans before they proceed.
8. **BitLesson Discipline**: Require running `bitlesson-selector` before each sub-task and record selected lesson IDs (or `NONE`) in the work notes.

### Important

- Use the Task tool to spawn agents as team members
- Monitor team members and reassign work if they get stuck
- Merge team work and resolve any conflicts before writing your summary
- Do NOT write code yourself - if you catch yourself about to edit a file or run implementation commands, delegate it instead
- When teammates go idle after sending you a message, this is NORMAL - they are waiting for your response, not done forever

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/root/miles/.humanize/rlcr/2026-06-24_06-04-55/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/root/miles/.humanize/rlcr/2026-06-24_06-04-55/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/root/miles/.humanize/rlcr/2026-06-24_06-04-55/round-0-summary.md
