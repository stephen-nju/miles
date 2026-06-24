# Round 0 Contract

## Round objective (overall)
Deliver the opt-in multi-process session server per the plan (M0→M5). A round only ends when the WHOLE plan is believed done; this contract keeps the round anchored to the mainline spine and prevents side-issue drift.

## Immediate focus for the first work block
The plan is a sequential dependency chain (M0→M1→M2→M3→{M4,M5}); little parallelism exists until M3. The first focus is the foundational spine that unblocks everything else:
- **M0 (task2)**: on `refactor/multi-process-session-server` (at `9cf2a0384`), re-establish the per-session in-flight gate (409) + strict upstream-response validation (502) cleanly, WITHOUT the bounded CPU thread pool. This fixes the single-process equivalence baseline AC-4 references.
- **M1 (task3)**: decouple the session handlers from FastAPI into transport-neutral core callables (create/get/delete/chat/proxy) with `SessionError`→status mapping, preserving query/header stripping; keep a thin FastAPI adapter + `do_proxy` facade so workers=1 and all existing HTTP tests run unchanged.

## Target ACs (immediate focus)
- AC-4 (behavioral-equivalence baseline established by M0; core decoupling preserves it in M1)
- AC-1 (opt-in / backward compatibility: workers=1 path stays byte-for-byte; adapter keeps existing suites green)

## Blocking side issues in scope
- None currently.

## Queued side issues OUT of scope this round
- RLCR base branch = `main` (should be `9cf2a0384` for the eventual `codex review`); only matters at Review Phase. Revisit before Review Phase; does not block coding.
- Consistent-hashing routing (Alt-3) and delta-R3 protocol — explicitly deferred per plan/DEC-4.

## Round success criteria
- Mainline tasks task1..task13 progressed along the M0→M5 chain.
- M0: gate(409)+validation(502) re-established without thread pool; `tests/fast/router/test_sessions.py` + a re-derived race/validation suite pass on the new branch.
- M1: existing public-HTTP suites pass unchanged via the adapter (workers=1).
- workers=1 remains byte-for-byte single-process (AC-1).
- No side issue replaced the mainline objective; goal-tracker kept current.

## Discipline
- Team Leader coordinates/delegates only; teammates own strict file boundaries; sequential file-touching tasks are ordered via dependencies to avoid silent overwrites.
- Run `bitlesson-selector` before each task; record lesson IDs (or NONE) in notes and the round summary's `## BitLesson Delta`.
