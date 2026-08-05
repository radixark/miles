#!/bin/bash
# Kimi-K2: 64 nodes, disaggregated.
# WARNING: Requires 512 GPUs. Do NOT run on clusters with fewer than 64 nodes.
#
# Usage: bash Kimi-K2.sh [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1..63 (workers)
#   HEAD_NODE_IP : IP address of the head node
set -ex

MODE="${1:-p2p}"
NODE_RANK="${2:-0}"
HEAD_NODE_IP="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (skip if checkpoint exists)
if [ ! -d /root/multinode/Kimi-K2-Instruct_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare Kimi-K2-Instruct
fi

# Run
python "${SCRIPT_DIR}/run.py" run Kimi-K2-Instruct \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
