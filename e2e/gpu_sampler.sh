#!/usr/bin/env bash
# Sample every GPU of this node for the run report: <out.csv> [seconds]. The harness runs it
# directly on the head node and as a Ray job pinned to every other node; teardown stops both.
set -uo pipefail
out=$1; secs=${2:-7200}
exec timeout "$secs" nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader -l 15 > "$out" 2>/dev/null
