#!/bin/bash

# Prepare script for GLM-4.5-Air (106B-A12B): download model/datasets and convert checkpoint.
#
# GLM-4.5-Air uses PP=4 for training, so checkpoint conversion requires a multinode
# Ray cluster.  The prepare step is split into two phases:
#   1. Download model + datasets  (all nodes)
#   2. Convert checkpoint          (head node only, with Ray cluster)
#
# Usage:
#   bash prepare-glm4.5-air.sh [--download-only]
#
# Options:
#   --download-only  Skip checkpoint conversion (for rollout/worker nodes)

set -ex

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DOWNLOAD_ONLY=0
for arg in "$@"; do
    if [ "$arg" = "--download-only" ]; then
        DOWNLOAD_ONLY=1
    fi
done

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="GLM-4.5-Air"
MODEL_TYPE="glm4.5-106B-A12B"
HF_REPO="zai-org/GLM-4.5-Air"
GPUS_PER_NODE=8

# Checkpoint save directory (override with CKPT_SAVE_DIR for shared storage)
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/root}"

echo "=== Preparing ${MODEL_NAME} ==="
echo "HF repo       : ${HF_REPO}"
echo "Model type    : ${MODEL_TYPE}"
echo "Ckpt dir      : ${CKPT_SAVE_DIR}"
echo "Download-only : ${DOWNLOAD_ONLY}"

# ---------------------------------------------------------------------------
# Download model and datasets
# ---------------------------------------------------------------------------
mkdir -p /root/models /root/datasets
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_REPO}', local_dir='/root/models/${MODEL_NAME}')
"

python3 -c "
from miles.utils.external_utils.command_utils import hf_download_dataset
hf_download_dataset('zhuzilin/dapo-math-17k')
hf_download_dataset('zhuzilin/aime-2024')
"

# ---------------------------------------------------------------------------
# Convert checkpoint to megatron format
# Skip if --download-only (e.g. worker/rollout node doesn't need megatron format)
# ---------------------------------------------------------------------------
if [ "${DOWNLOAD_ONLY}" -eq 0 ]; then
    python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${GPUS_PER_NODE},
    multinode=True,
    dir_dst='${CKPT_SAVE_DIR}',
)
"
    echo "Prepare done (full: download + convert)."
else
    echo "Prepare done (download-only: skipped checkpoint conversion)."
fi
