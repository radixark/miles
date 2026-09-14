#!/usr/bin/env bash
# Stop every process on this node whose environment carries the given line, e.g.
# MILES_PRESSURE_RUN=<run>: what a pressure run started here and Ray did not reap when its job
# stopped (the SGLang servers outlive their CommandActor). A prefix such as MILES_PRESSURE_RUN=
# sweeps the leftovers of every earlier run. Never touches this shell or its ancestors.
set -uo pipefail
pattern=${1:?environment line pattern, e.g. ^MILES_PRESSURE_RUN=run2-123$}
chain=" "; pid=$$
while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ]; do
    chain="$chain$pid "; pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
done
pids=""
for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    case "$chain" in *" $pid "*) continue ;; esac
    [ -r "$p/environ" ] || continue
    if { tr '\0' '\n' < "$p/environ" | grep -q -- "$pattern"; } 2>/dev/null; then pids="$pids $pid"; fi
done
if [ -z "${pids// /}" ]; then echo "[sweep $(hostname)] nothing matches $pattern"; exit 0; fi
echo "[sweep $(hostname)] stopping:$pids"
kill -TERM $pids 2>/dev/null; sleep 5; kill -KILL $pids 2>/dev/null; sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; echo
