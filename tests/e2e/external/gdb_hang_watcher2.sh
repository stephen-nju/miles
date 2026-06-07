#!/bin/bash
# Two-phase forensic watcher for the FT abort hang.
# Phase 1 (pre-hang): every 30s, snapshot py-spy --native of all GPU procs + the
#   latest training log line, so we can build a timeline of how the state evolves
#   into the deadlock (training -> crash -> reconfigure -> abort).
# Phase 2 (on 'aborting after'): intensive gdb capture, including
#   'thread apply all bt full' (locals/args -> lock addresses) so we can later
#   resolve WHO holds the mutex the stuck commFree/abortComms is waiting on.
# NOT a test; investigation helper (see agent-context 2026-06-08-*).
set -u
LOG="${1:-/tmp/pp2_hang_run3.log}"
OUT="${2:-/tmp/fx_capture}"
mkdir -p "$OUT"
echo "watcher armed at $(date -u +%H:%M:%S) watching $LOG" > "$OUT/meta.txt"

_pids() { pgrep -f 'MegatronTrainRayActor' 2>/dev/null | sort -un; }

# Phase 1: pre-hang snapshots (max ~45 min), stop once the timer abort fires.
for i in $(seq 1 90); do
    if grep -q "aborting after" "$LOG" 2>/dev/null; then
        echo "PHASE2 abort signature at $(date -u +%H:%M:%S) (poll $i)" >> "$OUT/meta.txt"
        break
    fi
    ts=$(date -u +%H%M%S)
    last=$(grep -aE 'rollout [0-9]|crash_before_allreduce|Reconfigured|allocate_for_pending|_mark_as' "$LOG" 2>/dev/null | tail -1 | cut -c1-160)
    pids=$(_pids)
    echo "pre i=$i ts=$ts npids=$(echo $pids|wc -w) log='${last}'" >> "$OUT/meta.txt"
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 25 py-spy dump --native --pid "$pid" > "$OUT/pre_${ts}_pid${pid}.txt" 2>&1 &
    done
    wait
    sleep 30
done

# Phase 2: intensive capture during the 600s abort window + watchdog SIGABRT.
for round in $(seq 1 12); do
    ts=$(date -u +%H%M%S)
    pids=$(_pids)
    echo "p2 round $round ts=$ts pids=[$(echo $pids|tr '\n' ' ')]" >> "$OUT/meta.txt"
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 25 py-spy dump --native --pid "$pid" > "$OUT/p2_pyspy_${ts}_pid${pid}.txt" 2>&1 &
    done
    wait
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 90 gdb -p "$pid" -batch \
            -ex "set pagination off" \
            -ex "thread apply all bt" \
            -ex "echo \n===BT FULL===\n" \
            -ex "thread apply all bt full" \
            > "$OUT/p2_gdb_${ts}_pid${pid}.txt" 2>&1
        # kernel-level per-thread stacks (shows futex/lock waits without symbols)
        for t in /proc/$pid/task/*/stack; do
            echo "--- $t ---"; cat "$t" 2>/dev/null
        done > "$OUT/p2_kstack_${ts}_pid${pid}.txt" 2>&1
    done
    sleep 25
done
echo "watcher done at $(date -u +%H:%M:%S)" >> "$OUT/meta.txt"
