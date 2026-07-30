#!/bin/bash
# GLM-4.5-Air: 8 nodes, disaggregated.
# Checkpoint conversion requires multinode Ray (PP=4), so head does full
# prepare while workers use --download-only.
#
# Usage: bash GLM-4.5-Air.sh [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1..7 (workers)
#   HEAD_NODE_IP : IP address of the head node
set -ex

MODE="${1:-p2p}"
NODE_RANK="${2:-0}"
HEAD_NODE_IP="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare: head does full convert, workers download-only
if [ "${NODE_RANK}" -eq 0 ]; then
    python "${SCRIPT_DIR}/run.py" prepare GLM-4.5-Air
else
    python "${SCRIPT_DIR}/run.py" prepare GLM-4.5-Air --download-only
fi

# Run
python "${SCRIPT_DIR}/run.py" run GLM-4.5-Air \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
