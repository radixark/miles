#!/bin/bash
# GLM-Z1-9B-0414: single-node, 4 train + 4 rollout GPUs.
# Usage: bash GLM-Z1-9B.sh [MODE]
#   MODE : p2p (default) | broadcast
set -ex

MODE="${1:-p2p}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepare (skip if checkpoint exists)
if [ ! -d /root/multinode/GLM-Z1-9B-0414_torch_dist ]; then
    python "${SCRIPT_DIR}/run.py" prepare GLM-Z1-9B-0414
fi

# Run
python "${SCRIPT_DIR}/run.py" run GLM-Z1-9B-0414 --mode "${MODE}"
