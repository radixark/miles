#!/bin/bash
# Qwen3-30B-A3B: 4 nodes, disaggregated.
# Usage: bash Qwen3-30B-A3B.sh [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1..3 (workers)
#   HEAD_NODE_IP : IP address of the head node
set -ex

MODE="${1:-p2p}"
NODE_RANK="${2:-0}"
HEAD_NODE_IP="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (skip if checkpoint exists)
if [ ! -d /root/multinode/Qwen3-30B-A3B_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare Qwen3-30B-A3B
fi

# Run
python "${SCRIPT_DIR}/run.py" run Qwen3-30B-A3B \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
