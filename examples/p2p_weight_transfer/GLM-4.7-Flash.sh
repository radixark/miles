#!/bin/bash
# GLM-4.7-Flash: 2 nodes, disaggregated.
# Usage: bash GLM-4.7-Flash.sh [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1 (worker)
#   HEAD_NODE_IP : IP address of the head node
set -ex

MODE="${1:-p2p}"
NODE_RANK="${2:-0}"
HEAD_NODE_IP="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (install custom transformers + convert checkpoint)
if [ ! -d /root/multinode/GLM-4.7-Flash_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare GLM-4.7-Flash
fi

# Run
python "${SCRIPT_DIR}/run.py" run GLM-4.7-Flash \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
