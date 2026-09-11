#!/usr/bin/env bash
# Stop every process on this node whose environment carries the given marker line
# (default: any MILES_E2E_RUN, i.e. anything a gateway e2e run started here). Used on the
# worker nodes of a cluster run, where the harness itself never executes.
set -uo pipefail
pattern=${1:-^MILES_E2E_RUN=}
pids=""
for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    if { tr '\0' '\n' < "$p/environ" | grep -q "$pattern"; } 2>/dev/null; then pids="$pids $pid"; fi
done
if [ -z "${pids// /}" ]; then echo "[sweep $(hostname)] nothing matches $pattern"; exit 0; fi
echo "[sweep $(hostname)] stopping:$pids"
kill -TERM $pids 2>/dev/null; sleep 8; kill -KILL $pids 2>/dev/null; sleep 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
