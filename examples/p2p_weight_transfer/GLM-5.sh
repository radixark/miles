#!/bin/bash
# GLM-5 variants: GLM-5_4layer | GLM-5_20layer | GLM-5
# Usage: bash GLM-5.sh <VARIANT> [MODE] [NODE_RANK] [HEAD_NODE_IP]
#   VARIANT      : GLM-5_4layer | GLM-5_20layer | GLM-5
#   MODE         : p2p (default) | broadcast
#   NODE_RANK    : 0 (head, default) | 1..N (workers)
#   HEAD_NODE_IP : IP address of the head node
set -ex

VARIANT="${1:?Usage: $0 <VARIANT> [MODE] [NODE_RANK] [HEAD_NODE_IP]}"
MODE="${2:-p2p}"
NODE_RANK="${3:-0}"
HEAD_NODE_IP="${4:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate variant
case "${VARIANT}" in
    GLM-5_4layer|GLM-5_20layer|GLM-5)
        ;;
    *)
        echo "ERROR: Unknown variant '${VARIANT}'. Use GLM-5_4layer, GLM-5_20layer, or GLM-5."
        exit 1
        ;;
esac

# Prepare
if [ "${NODE_RANK}" -eq 0 ]; then
    python "${SCRIPT_DIR}/run.py" prepare "${VARIANT}"
else
    python "${SCRIPT_DIR}/run.py" prepare "${VARIANT}" --download-only
fi

# Run
python "${SCRIPT_DIR}/run.py" run "${VARIANT}" \
    --mode "${MODE}" --node-rank "${NODE_RANK}" --head-ip "${HEAD_NODE_IP}"
