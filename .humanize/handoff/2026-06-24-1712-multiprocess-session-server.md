# Handoff — Opt-in multi-process session server (router + IPC → headless workers)

> Repo: `/root/miles` (radixark/miles) · Branch: `refactor/multi-process-session-server` · Base commit: `9cf2a0384`
> Written: 2026-06-24 17:12 · Author conversation: session `54be2c97-11ed-4ea3-86b0-a8326a72b093`
> **Status at handoff: feature is COMPLETE and fully pushed. The RLCR loop was CANCELLED by the user. The only open work is (a) the user's current ask "把现在的 wip 文档也 push 到远程" and (b) optionally opening the PR.**

---

## 1. Motivation and Success Picture

### The problem (user's own framing, gen-idea)
The current branch effort to make the session server faster via multi-threading / in-process concurrency is **futile because it is GIL-bound**. Under multi-turn agent rollouts, the session server's dominant cost is `json.loads` + pydantic validation of large **R3** (`routed_experts` / `indexer_topk`) response bodies — ~1 KB/token, so late-turn bodies reach **100+ MiB** (final-turn body measured at 134 MiB). A bounded thread pool cannot raise aggregate parse throughput (all threads serialize on the one GIL) and it *worsens* tail latency under concurrency.

### The idea (user's design, refined in conversation)
Split the session server into a **thin router process + N headless worker processes**, route each request by a stable hash of `session_id` (`worker_idx = stable_hash(session_id) % n_worker`), each worker internally serial-per-session. This works *specifically because sessions are nearly standalone* — there is no inter-process state-transfer cost, so sticky-by-hash means IPC carries only this-turn's request/response, never accumulated session state. Parsing the big R3 bodies now happens across N OS processes → N GILs → escapes the bottleneck.

### Two user corrections that are load-bearing (do NOT regress these)
1. **The router does inter-process (IPC) forwarding, NOT a second HTTP hop.** User: *"理论上 session server 的 router 应该非常轻，非常 tiny，目标是进程间做转发到对应 worker，而不是走 http 转发到 worker，这是一个根本区别。"* The inference-path `MilesRouter` is HTTP **because its targets are cross-GPU/cross-node engines**; the session router's targets are **same-host sibling processes**, a different goal → IPC (framed UNIX socket), not HTTP. Do not "reuse the existing proxy as a blueprint" if that means an HTTP hop.
2. **R3 is stripped from the client-facing chat response *uniformly* (both workers=1 and workers>1).** User confirmed DEC-1=B ("B 剥离(推荐)"). The agent never reads R3 from the chat response (it reads `choices[0].message`); R3 is needed only for training-sample reconstruction, which reads it from the stored record via `GET /sessions/{id}`.

### Success picture (aligned with user)
- Opt-in: `--session-server-workers=1` (default) = byte-for-byte today's single-process server **except** the one uniform R3 strip; spawns no router/extra processes.
- `>1` = thin router + N workers over IPC, behaviorally equivalent on all error/gate/passthrough/validation paths.
- **Measured, significant parallel-parse win** (DEC-4, directional): OLD-vs-NEW latency + throughput at p50/p95/**p99** + total wall time. (Benchmark task was explicitly added by the user mid-loop — port `bench_session_server_overhead.py` from `refactor/session-server-concurrency`, extend to multi-CPU-worker input, measure the deltas.)

### Explicit scope exclusions (do NOT expand)
- **Consistent-hashing / rebalanceable routing** (Alt-3): v1 uses simple modulo over a stable hash. Deferred.
- **Delta / incremental-R3 protocol** ("每轮只返回增量 r3"): user said *"目前做不了"* — needs SGLang/trainer coordination outside this repo. Deferred.
- Do **not** rebuild the bounded CPU thread pool (DEC-0 explicitly dropped it; multi-process replaces it).

### Standing user instruction (applies to all future code changes)
*"每当你改完代码都要 push 远程 upstream，防止代码丢失"* — push to the `upstream` remote after every code change to prevent loss.

---

## 2. Key Decisions and Alternatives

| ID | Decision | Why | Rejected alternative (and why) |
|----|----------|-----|--------------------------------|
| **DEC-0** | Branch from `9cf2a0384` (NOT `refactor/session-server-concurrency`); re-establish the per-session in-flight gate (409 `SessionBusyError`) + strict upstream-response validation (502 `UpstreamResponseError`) **inline**; **drop** the bounded CPU thread pool. | The concurrency branch's thread pool is the GIL-bound dead end this whole effort replaces. The `9cf2a0384` baseline was confirmed (Codex + git) to *lack* the gate/validation/pool, so they had to be re-derived as the workers=1 equivalence baseline. | Keep the thread pool → rejected: GIL-bound, cannot raise aggregate parse throughput, worsens tail latency. Build on the concurrency branch → rejected by user (start clean from 9cf2a0384). |
| **(router transport)** | Router forwards via **framed UNIX-socket IPC**, never a 2nd HTTP hop; router never `json.loads` the chat body. | Targets are same-host sibling processes; an HTTP hop would re-incur serialize/parse of the 100+ MiB body in the router — relocating the bottleneck. | Model on `MilesRouter` (HTTP) → rejected by user: MilesRouter is HTTP only because *its* targets are cross-GPU/node engines; different goal. |
| **DEC-1 = B** | Strip R3 (`routed_experts`/`indexer_topk`) from the **client chat response** uniformly (both modes); keep full R3 in the worker's stored record; serve it via `GET /sessions/{id}`. | The client never reads R3 from the chat response; parsing/re-serializing 100+ MiB for the client is the dominant avoidable cost; uniform avoids a worker-count-dependent API shape (a semantic smell that would break custom agents only at scale). | A) Keep returning R3 in the client chat response → rejected (the cost we're removing). Worker-count-dependent strip → rejected (smell). |
| **DEC-2** | Any worker death → **whole session server fail-fast**, surfaced to the rollout path. | A silently-restarted worker returns *fresh empty state* (200) for a pre-existing session = silent training-data corruption. | Silent per-worker restart → rejected (masks data loss). |
| **DEC-3** | Firm 32-hex lowercase session id; routing uses process-stable `blake2b(session_id) % n_worker`. | Routing must be identical across process restarts; Python builtin `hash()` is `PYTHONHASHSEED`-salted → would remap ids across processes. | builtin `hash()` → rejected (per-process salt breaks cross-process determinism). |
| **DEC-4** | Success = **directional** parallel-parse win (latency/throughput/wall measured); delta-R3 deferred. | User accepted "架构 + 并行解析收益(方向性)" as the bar. | Require a full delta-R3 protocol → deferred (cross-repo coordination). |
| **DEC-5** | IPC transport = framed UNIX-socket multiplex (request-id-keyed, out-of-order replies, single-writer-per-socket, round-robin chunking, bounded send buffer, EOF→fail-pending-futures). | Gated by AC-5/AC-6: must overlap distinct-session I/O, bound CPU parse, and not head-of-line-block large bodies. | (transport was an allowed design choice; this shape passed the pre-impl adversarial design review). |

Process/lifecycle decisions baked into the supervisor (from the pre-impl design review, `m3-design-contract.md`): multiprocessing **spawn** ctx; socketpair-per-worker; **non-daemon** children (daemons can't spawn their own children); `PR_SET_PDEATHSIG` so workers die with the parent; fail-fast **process-group** teardown; `asyncio.shield` on the router so a client disconnect does not cancel the in-flight worker task.

---

## 3. Approaches Tried but Failed / Discarded (pits — do not re-step)

1. **Assuming the `9cf2a0384` baseline already had the gate/validation/thread-pool.** It did NOT (confirmed via Codex review + `git show`). Fix: added milestone **M0** (`f2b4a5e39`) to re-establish gate+validation inline. Don't assume baseline parity — it's a *pre-feature* commit.
2. **Two background teammates editing one shared working tree → collisions.** A duplicate teammate was spawned on a *false* "teammate died" inference (a waiter exited 144 + a `pgrep` miss), producing stray uncommitted instrumentation in `session_worker.py`. Fix: `git restore --source=HEAD <file>`, then guard every teammate commit with explicit `git add <files>` + `git diff --cached --stat`. → BitLesson **BL-20260624-agent-teams-shared-tree**. Lesson: never run overlapping teammates in a shared tree (use worktree isolation or strict sequencing); never infer teammate death from a waiter/pgrep signal.
3. **Amending already-pushed commits** (m5-equivalence amended `aad36a025`→`c1485878d`→`ce0b96b3f`) forced `git push --force-with-lease`. Avoid amending pushed commits; add follow-up commits instead.
4. **Benchmark "worker death" at ~248/1600 turns was NOT a code defect.** Two independent investigations (~11k+ large-body turns at exact-config + harder w8 stress, 0 deaths; plus a structural review showing every worker exception path is contained) root-caused it to a **benchmark keep-alive connection-reset cascade**: the bench client out-waited uvicorn's 5 s keep-alive during the OLD single-process GIL stall; the reset tripped the supervisor's *by-design* fail-fast teardown, which *looked* like a crash (no traceback, not OOM, not fd). Fixed **entirely harness-side** (mock `timeout_keep_alive`, client `keepalive_expiry` + idempotent retry, opt-in big GET). **No session source changed for it.** → BitLesson **BL-20260624-gil-server-bench-keepalive**. Don't go hunting for a worker crash bug here — it's a benchmarking artifact.
5. **Stop-hook deadlock (process, not code).** The TodoWrite MCP disconnected mid-session, freezing an incomplete TodoWrite snapshot in the transcript that the loop's stop-hook checker (`check-todos-from-transcript.py`, union of Task-dir + latest-TodoWrite) reads → the loop could not be exited cleanly. Re-creating+completing Task tasks did NOT clear the TodoWrite-source blockers (verified empirically by running the checker). Editing the transcript/state was refused (the loop's no-cheating rule: only the USER may cancel). Resolved only when the user ran `/humanize:cancel-rlcr-loop`. The loop is now **CANCELLED** (`.humanize/rlcr/2026-06-24_06-04-55/cancel-state.md` + `.cancel-requested` exist). Do not try to restart/resume the loop unless the user asks.

---

## 4. Current Code / File State

**Working tree is CLEAN. All 10 commits are pushed and in sync** with `upstream/refactor/multi-process-session-server` (`git rev-list --left-right --count upstream/...HEAD` → `0  0`).

```
$ git status --porcelain
(empty)
$ git log --oneline 9cf2a0384..HEAD
da1a2913c fix(session): address Codex P1/P2 review findings in multi-process server
ce0b96b3f test(session): AC-4 multi-process equivalence (workers=N error/gate/passthrough)
1b338d7ad test(session): pin client-disconnect-does-not-destabilize invariant + bench summary
1162e8186 bench(session): HTTP end-to-end mode for OLD vs NEW session-server overhead
f0a9c4808 feat(session): thin router process + multi-process supervisor (workers>1)
2ae9525ee fix(session): map create_session_with_id ValueError to explicit 400/409
606dcb1b5 feat(session): data plane — uniform R3 strip, IPC channel, headless worker
fffa0a6ad feat(session): M2 scaffolding for multi-process session server
202113e5e refactor(session): decouple session handlers into a transport-neutral core
f2b4a5e39 feat(session): per-session in-flight gate + strict upstream validation (inline)
```

### Source files (all committed)
| File | State | Notes |
|------|-------|-------|
| `miles/rollout/session/session_core.py` | NEW (M1) | Transport-neutral `SessionCore`, `CoreResponse`, `ProxyRequest`; `SessionError`→status in-core; `create_session_with_id` (ValueError→400/409); `_strip_r3_from_choice` (ALL choices); GET offloads `model_dump()`/`json.dumps()` via `asyncio.to_thread`. |
| `miles/rollout/session/session_ipc.py` | NEW (M3-A) | Framed multiplexed RPC; round-robin writer (`_OutboundBody`, one chunk/active body), `DEFAULT_MAX_SEND_BUFFER_BYTES`=256 MiB; `IpcChannel`, `open_unix_channel`, `encode_envelope`/`decode_envelope`. |
| `miles/rollout/session/session_worker.py` | NEW | `run_worker(args, backend_url, sock, worker_index)` (Process target), `SessionWorker`, `ProxyBackend`, parse `asyncio.Semaphore`, `_set_pdeathsig`. |
| `miles/rollout/session/session_router.py` | NEW (M3-B) | `SessionRouter`, `run_router`; route order chat-before-catch-all; never `json.loads`; `asyncio.create_task` + `_ipc_tasks` drain set; `_inflight` decremented in done-callback; 503 backpressure. |
| `miles/rollout/session/session_supervisor.py` | NEW | `SessionServerSupervisor.start()/.shutdown()/.check()`; signal handler restores `SIG_DFL` + `os.kill(os.getpid(), signum)`. |
| `miles/rollout/session/routing.py` | NEW | `worker_index_for_session` (blake2b), `new_session_id`. |
| `miles/rollout/session/sessions.py` | modified | Thin FastAPI adapter over `SessionCore` (workers=1 path); `do_proxy` facade kept patch-compatible. |
| `miles/rollout/session/session_server.py`, `linear_trajectory.py`, `session_errors.py` | modified | Supporting wiring for the core decouple + gate/validation. |
| `miles/ray/rollout/router_manager.py` | modified | `start_session_server` returns the supervisor for workers>1; workers=1 path unchanged. |
| `miles/ray/rollout/rollout_manager.py` | modified | Stores `self.session_server_supervisor`; `_check_session_server()` called in `generate()` / `eval()` (fail-fast propagation). |
| `miles/utils/arguments.py` | modified | `--session-server-workers` (int, default 1). |

### Test files (all committed; `tests/fast/router/` → 144 passing, no orphans)
`test_session_multiprocess.py`, `test_session_multiprocess_equivalence.py`, `test_session_multiprocess_disconnect.py`, `test_session_ipc.py`, `test_session_worker.py`, `test_session_routing.py`, `test_session_router_backpressure.py` (+ updated `test_session_race_conditions.py`). The race suite stays the workers=1 proof (its in-process `patch.object` failure injection can't cross the spawn boundary); equivalence tests cover the client/backend-driven paths that *do* cross the boundary at workers=N.

### Benchmark (committed)
`tests/benchmark/bench_session_server_overhead.py` (HTTP mode: `--mode http --session-server-workers N`), `tests/benchmark/_mock_r3_backend.py`, `tests/benchmark/results/*.json` (6 configs), `tests/benchmark/results/SUMMARY.md`, `tests/benchmark/results/.gitignore` (ignores `*.log`).

### Gitignored WIP docs (NOT tracked — `.gitignore:196` ignores `.humanize/`)
These are the "WIP 文档" the user's current ask refers to. They are **not** on any branch and cannot ride the feature branch without `git add -f` (which would pollute the PR). See §6 for the open push question.
- `.humanize/ideas/session-server-20260624-030743.md` (+ overhead idea drafts)
- `.humanize/plans/session-server-20260624-030743-20260624-034515.md` (+ `_zh`)
- `.humanize/rlcr/2026-06-24_06-04-55/`: `goal-tracker.md`, `round-0-contract.md`, `m3-design-contract.md`, `round-0-summary.md`, `plan.md`, `cancel-state.md`, `.cancel-requested`
- `.humanize/bitlesson.md` (2 lessons added: BL-20260624-agent-teams-shared-tree, BL-20260624-gil-server-bench-keepalive)

### Scratchpad (outside repo, in `/tmp`)
- `…/scratchpad/PR_DESCRIPTION.md` — drafted, ready PR body (also see §9).
- `…/scratchpad/codex-final-codereview.md` — the final Codex review prompt.

---

## 5. Current Plan: Done / In-Progress / Remaining

### Done (all tasks task1–task13; ACs AC-1…AC-7)
- **M0** (`f2b4a5e39`): inline in-flight gate (409) + strict upstream validation (502), no thread pool.
- **M1** (`202113e5e`): transport-neutral `SessionCore` + thin FastAPI adapter (workers=1 unchanged).
- **M2** (`fffa0a6ad`, `2ae9525ee`): `--session-server-workers`; `create_session_with_id` (32-hex, no overwrite, ValueError→400/409); `routing.py` (blake2b).
- **M3 data plane** (`606dcb1b5`): `session_ipc.py`, `session_worker.py`, uniform R3 strip.
- **M3 control plane / M4** (`f0a9c4808`): `session_router.py`, `session_supervisor.py`, `router_manager` wiring.
- **M5** (`1162e8186`, `1b338d7ad`, `ce0b96b3f`): HTTP-mode benchmark + `_mock_r3_backend.py`; workers=N equivalence tests; disconnect-stability tests; SUMMARY.md.
- **Codex review fixes** (`da1a2913c`): all 5 P1 + 2 P2 fixed (fail-fast→rollout propagation; SIGTERM restores SIG_DFL + re-raises; disconnect runs shielded IPC as a tracked task + decrements backpressure in done-callback; IPC writer round-robins one chunk/body with bounded queued-bytes; R3 stripped from ALL choices; large GET-records offloaded via `asyncio.to_thread`). P1-E (workers=1 JSON canonicalization) investigated + documented as accepted (raw byte passthrough; agents consume parsed content).

**Verification on record:** `python -m pytest tests/fast/router/ -q` → **144 passed**, 0 failures, no orphan `miles-session-*` processes. Benchmark (32 sessions × 50 turns × r3_scale=1000): OLD w1 vs NEW w16 = **6.7×** wall/throughput, p50 **10.3×** / p95 **8.9×** / p99 **8.2×**, ~flat RSS; w16 is the sweet spot (w32 over-subscribes). See §9 for the full metrics table.

### In-progress
- **The user's current ask: "把现在的 wip 文档也 push 到远程"** — push the gitignored `.humanize/` WIP docs (+ this handoff doc) to a remote. BLOCKED on the §6 decision (which remote / which branch), because `.humanize/` is gitignored and must NOT pollute the feature-branch PR. Stopped here to ask the user; do not force-add onto `refactor/multi-process-session-server`.

### Remaining (in dependency order)
1. **Push WIP docs to remote** (current ask). Acceptance: `.humanize/` docs + this handoff doc are retrievable from a remote ref; the feature branch / PR diff is unchanged. Recommended mechanism: a dedicated `wip-docs/session-server` branch with the docs `git add -f`'d, pushed to the user's chosen remote — NOT committed onto `refactor/multi-process-session-server`.
2. **(Optional, user-gated) Open the PR.** Acceptance: `gh pr create` against the correct base using `…/scratchpad/PR_DESCRIPTION.md`. **Outward-facing — requires explicit user confirmation.** Note the base-branch caveat in §6.

---

## 6. Open Questions / Blockers

1. **Where to push the WIP docs? (current ask — blocks Remaining #1.)** `.humanize/` is gitignored (`.gitignore:196`), so it can't go on the feature branch without polluting the PR. Resolver: **user**. Options: (a) **[recommended]** a dedicated `wip-docs/session-server` branch, docs `git add -f`'d, pushed to the user's personal fork `myrepo` (`git@github.com:guapisolo/miles.git`) — keeps the org `upstream` clean; (b) same dedicated branch but pushed to `upstream` (radixark — matches the standing "push to upstream" habit, but adds a docs branch to the org repo); (c) skip the repo push, keep docs only in `/tmp`. Until answered, do not push.
2. **Open the PR now, or wait?** Resolver: **user** (outward-facing). PR body is drafted (§9).
3. **PR/review base-branch caveat.** The branch was cut from `9cf2a0384`, but `main` has since diverged (e.g. `1daf70714`, `c140c4769` are on main but not in this branch's lineage). `codex review --base main` / a PR diff against `main` would show unrelated divergence. Decide the review/PR base deliberately — diff against the branch point `9cf2a0384` for a clean feature diff, or rebase onto current `main` first if the PR must target `main`. Resolver: **user/picker-upper**.
4. **RLCR loop is CANCELLED — do not resume it.** The stop-hook deadlock (see §3.5) was a TodoWrite-MCP/transcript artifact, not unfinished work. The feature itself is done. Resolver: only the user may re-start a loop.

---

## 7. Related Skills (call order when picking up)

1. **`motivation-contract`** — first, to re-anchor "why + what counts as done". The IMMUTABLE goal + AC-1…AC-7 live in `.humanize/rlcr/2026-06-24_06-04-55/goal-tracker.md`; read that to reconstruct intent without re-asking. Needed because the next agent must not regress the two load-bearing corrections (IPC-not-HTTP; uniform R3 strip).
2. **`commit`** — the main drive-forward path now, because the *code* feature is complete and committed; the remaining work is git/remote plumbing (the WIP-docs push) and an optional PR. Use it to shape the WIP-docs branch + push cleanly and (if the user approves) to open the PR. NOT `normal-dev`/`debug-fix`/`refactor-*` — there is no remaining implementation.
3. **`subagent-harness`** — only if the user wants *another* independent code review before merge (spawn an independent reviewer per this skill; map it to the harness's own agent mechanism). The prior Codex review (`ask-codex.sh`, gpt-5.5:xhigh) already ran and all findings are fixed; a fresh review is optional, not required.
4. Do **NOT** invoke `start-rlcr-loop` / `cancel-rlcr-loop` — the loop is cancelled and the feature is done; re-entering the loop would re-trigger the stop-hook artifact.

> Note: any `commit`/push must keep the standing rule (commit messages end with the `Co-Authored-By:` + `Claude-Session:` trailers) and push code to `upstream`. The WIP-docs push target is the §6.1 question.

---

## 8. Verify-Before-Resume Checklist

Run these before touching anything; if any disagrees with §4, STOP and reconcile.

- [ ] `cd /root/miles && git rev-parse --abbrev-ref HEAD` → `refactor/multi-process-session-server`.
- [ ] `git status --porcelain` → empty (clean tree). If dirty, something changed since handoff — investigate before pushing.
- [ ] `git rev-list --left-right --count upstream/refactor/multi-process-session-server...HEAD` → `0  0` (in sync). If not, the feature branch moved.
- [ ] `git log --oneline 9cf2a0384..HEAD` → the 10 commits in §4 (top = `da1a2913c`).
- [ ] `git check-ignore -v .humanize` → confirms `.humanize/` is still gitignored (the §6.1 reason). If it became tracked, re-plan the push.
- [ ] Confirm the §1 motivation still holds — the feature is done; the only live ask is the WIP-docs push. The user may have changed scope.
- [ ] (If re-running tests) `python -m pytest tests/fast/router/ -q` from repo root → expect 144 passed, no orphan `miles-session-*` processes (~165 s, leader-run). This needs the project's Python env (tokenizer/TITO init); it is CPU/process-heavy.
- [ ] (If touching the benchmark) it needs a mock backend; see `tests/benchmark/_mock_r3_backend.py` and `results/SUMMARY.md` for the exact invocation.

---

## 9. Key References / Resources

### Full benchmark metrics (from `tests/benchmark/results/*.json`)
Workload (all configs identical): **32 sessions × 50 turns = 1600 turns**; 1000 in + 1000 out tok/turn; `r3_scale=1000` B/tok → ~76.5 GiB total R3 transferred, ~134 MiB final-turn body, ~5.94 GiB retained after pruning. Mock backend isolates session-server CPU/IPC overhead (not model latency).

| Config | Wall | Throughput | reply p50 | p95 | p99 | max | Peak RSS |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPU floor (serial, no HTTP) | 145.8 s | 10.97 t/s | 89 ms | 178 ms | 190 ms | 1265 ms | — |
| OLD w1 | 405.3 s | 3.95 t/s | 7763 ms | 16224 ms | 18477 ms | 19615 ms | 43.6 GiB |
| NEW w16 | 60.7 s | 26.37 t/s | 750 ms | 1819 ms | 2262 ms | 2343 ms | 44.9 GiB |
| NEW w32 | 71.8 s | 22.27 t/s | 712 ms | 2138 ms | 2514 ms | 2831 ms | 56.3 GiB |
| OLD w1 +get-records | 473.7 s | 3.38 t/s | (reply p99 44469) | — | — | 77100 ms | 52.2 GiB |
| NEW w32 +get-records | 63.3 s | 25.26 t/s | (reply p99 1898) | — | — | 7667 ms | 60.8 GiB |

- Per-stage CPU decomposition (from the serial floor): `response_parse_validate` p50 **79 ms** vs tokenization p50 4 ms — i.e. the R3 `json.loads`/validate is ~20× tokenization and dominates; this is the GIL-bound cost multi-process parallelizes.
- GET-records (big ~190 MiB read): OLD GET p50 39249 ms vs NEW p50 8063 ms (~4.8×); and in OLD the big GET blocks the single loop so other sessions' chat p99 explodes to 44469 ms (max 77100), whereas NEW isolates it (other sessions stay ~1.9 s p99).
- **w16 > w32**: with the per-worker parse-gate at 2, 16 workers already give 32 parse slots = the 32 concurrent sessions; w32 just adds per-process overhead + tokenizer-copy RSS. The right N tracks concurrent-session count, not core count.

### Document paths (all under `/root/miles`, gitignored)
- Goal tracker (IMMUTABLE goal + AC-1…AC-7 + decisions + deferrals): `.humanize/rlcr/2026-06-24_06-04-55/goal-tracker.md`
- Round-0 summary (what was built, validation, BitLesson delta): `.humanize/rlcr/2026-06-24_06-04-55/round-0-summary.md`
- M3 design contract (9 pre-impl must-fixes folded into the IPC/concurrency build): `.humanize/rlcr/2026-06-24_06-04-55/m3-design-contract.md`
- Round-0 contract: `.humanize/rlcr/2026-06-24_06-04-55/round-0-contract.md`
- Plan (+ Chinese): `.humanize/plans/session-server-20260624-030743-20260624-034515.md` (`_zh`)
- BitLesson: `.humanize/bitlesson.md`
- Loop cancel state: `.humanize/rlcr/2026-06-24_06-04-55/cancel-state.md` (+ `.cancel-requested`)
- Benchmark summary: `tests/benchmark/results/SUMMARY.md` (committed)

### Scratchpad (`/tmp/claude-0/-root-miles/54be2c97-11ed-4ea3-86b0-a8326a72b093/scratchpad/`)
- `PR_DESCRIPTION.md` — ready PR body. Headline: opt-in multi-process mode; 6.7× wall/throughput, 8–10× tail latency; 144 tests; one intended client-visible change (R3 strip); deferred = consistent-hashing + delta-R3.
- `codex-final-codereview.md` — the adversarial review prompt used (gpt-5.5:xhigh via `/root/humanize/scripts/ask-codex.sh`).

### Remotes
- `upstream` / `origin` = `radixark/miles` (org; code pushes go here per standing instruction).
- `myrepo` = `git@github.com:guapisolo/miles.git` (user's personal fork; recommended home for WIP docs).
- `prod` = `radixark/miles-prod`; `zyz` = `zyzshishui/miles`.

### Full prior transcript (if deeper detail needed)
`/root/.claude/projects/-root-miles/54be2c97-11ed-4ea3-86b0-a8326a72b093.jsonl`

---

**This handoff doc:** `/tmp/handoff/4d20d2c8-miles/2026-06-24-1712-multiprocess-session-server.md`
