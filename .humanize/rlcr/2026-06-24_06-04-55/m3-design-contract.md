# M3/M4 Implementation Contract (binding for tasks #3-#7)

Derived from the Codex pre-implementation adversarial review (`.humanize/skill/2026-06-24_06-42-54-*/output.md`) + the resolved R3-strip decision. Every item below is a REQUIREMENT, not a suggestion.

## Resolved decision — R3 strip is UNIFORM (workers=1 AND workers>1)
- `SessionCore.chat_completions` ALWAYS strips `routed_experts` / `indexer_topk` from the CLIENT-facing response body, in both single-process and multi-process modes. Build the client body by re-serializing the already-parsed response dict MINUS those two keys (no re-parse; R3 already removed → small body). The STORED `SessionRecord` keeps the FULL response (with R3).
- `GET /sessions/{id}` returns records WITH full R3 (unchanged) — that is how training-sample reconstruction (`compute_samples_from_openai_records` → `get_routed_experts_from_response`) gets R3.
- Update the meta_info byte-for-byte passthrough assertion in `tests/fast/router/test_session_race_conditions.py`: assert the client chat response preserves message + logprob meta but OMITS routed_experts/indexer_topk, AND assert a separate check that GET-records still contains them. AC-1 now means "byte-for-byte EXCEPT the uniform R3 strip" (logged in goal-tracker Plan Evolution Log).
- Net effect: the IPC path needs NO special R3 logic — the worker relays whatever SessionCore returns (already stripped for the client body); the big blob only crosses IPC on the infrequent GET-records path.

## IPC transport requirements
1. **Per-socket single-writer**: all sends on a given socket are serialized through ONE writer (asyncio queue or lock). Concurrent reply tasks MUST NOT interleave length-prefix + payload. Nonblocking sends.
2. **No HOL blocking from large bodies**: a large GET-records reply (full R3, possibly 100+ MiB) MUST NOT monopolize the per-worker stream and block small replies. Use a multiplexed framing protocol: every frame tagged by `request_id`, bounded max chunk size; large bodies split into multiple interleaved chunk-frames, reassembled per request_id at the reader. (Alternatively a separate bulk data channel per worker — implementer's choice, but small replies must not wait behind a big one.)
3. **Frame/body size cap**: enforce a max frame size and a max body size; exceed → deterministic error, not unbounded buffering.
4. **Reader robustness**: the per-worker reader has a deterministic failure path on EOF / partial frame / corrupt length. On EOF (worker death) it MUST fail all that worker's pending request_id futures AND trigger global fail-fast.
5. **Late/abandoned replies**: a reply arriving for an already-removed/cancelled request_id is dropped cleanly (no `InvalidStateError` that kills the reader).

## Spawn / fd ownership
6. Parent creates one channel per worker; passes ONLY the router-end to the router and ONLY the matching worker-end to each worker. All other ends/fds are closed in each process (parent, router, workers) so EOF is observable. Use multiprocessing spawn context. Validate this with a subprocess-leak test.

## Concurrency model
7. Each worker runs its own asyncio loop; each inbound IPC request is handled as its own task so different sessions' upstream `await`s overlap. The per-session in-flight gate stays in `SessionCore` (claim under `session.lock`, release once in `finally`).
8. **Parse-bound semaphore**: bound concurrent CPU parse/validate with a process-local `asyncio.Semaphore` (size ~1-2) for MEMORY bounding. It MUST NOT be awaited while holding `session.lock`, and the busy-409 fast-fail MUST happen BEFORE any wait on this semaphore (a same-session contender must still 409 quickly, not queue behind a parse). Accept that a big inline parse blocks that one worker's loop (the multi-process win is cross-worker parallelism, not intra-worker).
9. **Per-worker backpressure**: cap max in-flight requests and max queued bytes per worker; on exceed return a clear 503/429 (do not grow unbounded across router futures + frames + worker tasks + httpx queue).

## Router
10. Sole HTTP listener (FastAPI/uvicorn). Route order: the chat route `/sessions/{id}/v1/chat/completions` MUST be registered BEFORE the catch-all `/sessions/{id}/{path}` (else chat goes through proxy and skips record-keeping). Catch-all keeps NO-404 proxy semantics (route by session_id, do NOT call get_session / do NOT 404 on unknown id).
11. Router NEVER `json.loads` the chat body NOR the GET-records body — relay reply bytes in both cases.
12. `POST /sessions` → 200; router calls `routing.new_session_id()`, owner = `worker_index_for_session(id, N)`, sends a create-with-id op; needs a CORE-level create-with-id op on `SessionCore` (it currently lacks one — add it, mapping SessionError→status).
13. `/health` preserves at least `{"status":"ok","session_server_instance_id":...}` (read by `openai_endpoint_utils.py`); router reports healthy iff all workers alive; health pings have a timeout (a worker mid-parse may be briefly unresponsive).
14. **Cancellation**: the router handler `await`s the reply via `asyncio.shield` (or an abandoned-future design) so a client disconnect / handler cancel does NOT cancel the worker's chat task (today's semantics: the handler continues and may commit the turn — race tests rely on this). After disconnect the router still drains the IPC reply (so the worker never blocks on send).

## Lifecycle / fail-fast (M4)
15. Non-daemon children (daemon can't spawn children). Readiness = ALL workers' tokenizer/TITO init done (not just a TCP port) before `start_session_server` returns / router serves healthy.
16. Real fail-fast: a monitor that, on any child death, fails pending futures, kills the whole process GROUP, and surfaces the failure to the rollout/Ray-actor main path (a thread `raise` alone does NOT propagate). Add Linux `prctl(PR_SET_PDEATHSIG, SIGKILL)` (or a parent-death watchdog) so workers die if the parent dies. Process-group kill on SIGTERM/atexit/crash → no orphans.

## Optional (do if cheap)
- `--session-server-workers` arg validation `>=1`.
- A lightweight `ProxyBackend` for the worker instead of constructing a full `SessionServer`/FastAPI app (the worker only needs `do_proxy` + the httpx client).
- Per-worker httpx connection limit so N workers don't multiply the upstream cap to N*1024.

## Verification
- workers=1: `tests/fast/router/` all pass (with the updated R3 assertion).
- workers=N smoke/integration: create→chat→GET→DELETE through the router with N=2 workers on a mock backend; concurrent distinct-session chats overlap; a same-session second chat → 409 without backend; large GET-records concurrent with a small DELETE/health does not stall; worker-death triggers fail-fast; subprocess-leak check after shutdown.
