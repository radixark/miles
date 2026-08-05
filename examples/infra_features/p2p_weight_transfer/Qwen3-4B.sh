#!/bin/bash
# Qwen3-4B: single-node, 4 train + 4 rollout GPUs.
# Usage: bash Qwen3-4B.sh [MODE]
#   MODE : p2p (default) | broadcast
set -ex

MODE="${1:-p2p}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (skip if checkpoint exists)
if [ ! -d /root/multinode/Qwen3-4B_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare Qwen3-4B
fi

# Run
python "${SCRIPT_DIR}/run.py" run Qwen3-4B --mode "${MODE}"
