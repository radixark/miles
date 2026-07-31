#!/bin/bash
# Moonlight-16B-A3B-Instruct: 2 nodes, disaggregated.
# Usage: bash Moonlight-16B.sh [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1 (worker)
#   HEAD_NODE_IP : IP address of the head node
set -ex

MODE="${1:-p2p}"
NODE_RANK="${2:-0}"
HEAD_NODE_IP="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (skip if checkpoint exists)
if [ ! -d /root/multinode/Moonlight-16B-A3B-Instruct_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare Moonlight-16B-A3B-Instruct
fi

# Run
python "${SCRIPT_DIR}/run.py" run Moonlight-16B-A3B-Instruct \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
