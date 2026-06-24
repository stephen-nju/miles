# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: agent-teams-shared-tree
Lesson ID: BL-20260624-agent-teams-shared-tree
Scope: Agent Teams / RLCR multi-agent coordination on a single git working tree
Problem Description: Running multiple background teammates that edit the same working tree caused repeated collisions: (a) a duplicate teammate was spawned after wrongly inferring the original had died from a background-waiter abnormal exit + a `pgrep` miss; (b) two teammates left overlapping uncommitted edits in `session_worker.py` requiring repeated reverts; (c) a teammate amended an already-pushed commit, forcing a `--force-with-lease` to reconcile the remote.
Root Cause: Subagents share one checkout; there is no isolation, and teammate liveness cannot be reliably inferred from waiter/pgrep signals (process names don't match the agent; benches between runs look idle).
Solution: Never run overlapping teammates in the same working tree — sequence them strictly (one active editor) or use `isolation: "worktree"`. Never infer teammate death from a waiter exit or pgrep miss; rely on the teammate's own report/idle signal. Guard every commit with explicit `git add <paths>` (never `-A`) + verify `git diff --cached --stat` before committing. Revert stray uncommitted edits only after confirming the editing teammate has terminated.
Constraints: Worktree isolation adds merge-back overhead; for sequential single-task work, strict sequencing is simpler.
Validation Evidence: refactor/multi-process-session-server round 0 — collisions were detected and cleaned (tree ended clean, `git diff 9cf2a0384..HEAD -- miles/rollout/session/` exactly the intended modules; 137 fast/router tests green; 9 commits pushed).
Source Rounds: 0

## Lesson: gil-server-bench-keepalive
Lesson ID: BL-20260624-gil-server-bench-keepalive
Scope: Benchmarking a GIL-bound single-process HTTP server (session server) with large response bodies
Problem Description: An HTTP benchmark of the OLD single-process session server intermittently produced `httpx.ReadError`, and in multi-process mode a worker appeared to "die" ~248/1600 turns in (503 `IpcChannelClosed`) — with no traceback, not OOM, not fd exhaustion — which looked like a worker/IPC defect.
Root Cause: While the single event loop parses a 100+ MiB body it is blocked for seconds; a pooled client connection out-waits the server's 5s uvicorn keep-alive and resets on reuse. In multi-process mode that client-side reset propagated to a disconnect that tripped the supervisor's by-design fail-fast group teardown — a worker SIGKILLed by teardown has no Python traceback, exactly matching the misleading "silent worker death" signature. Not a code defect.
Solution: Benchmark client must not reuse keep-alive connections that can out-wait the server during a parse stall — set a short `keepalive_expiry` (< server keep-alive) and/or fresh connections, plus an idempotent-request retry; raise the mock/server `timeout_keep_alive` above the worst-case stall. To disambiguate a genuine worker crash from a teardown kill, arm opt-in per-worker crash-traceback capture (env-gated): a real crash leaves a traceback file, a teardown kill leaves none.
Constraints: Mock-backend benchmark isolates session-server CPU/IPC overhead, not end-to-end model latency; the gain scales with concurrency × body size.
Validation Evidence: After harness fixes, OLD w1 and NEW w16/w32/get-records all completed 1600/1600 turns, 0 transport/5xx; ~11k+ large-body turns across repros produced 0 worker deaths / 0 crash files. Headline NEW w16 vs OLD w1 = 6.7× throughput, 8–10× lower reply latency.
Source Rounds: 0
