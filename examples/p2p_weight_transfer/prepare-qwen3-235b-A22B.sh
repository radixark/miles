
#!/bin/bash

# Prepare script for Qwen3-235B-A22B: download model/datasets and convert checkpoint.
#
# Run this BEFORE run-qwen3-235B-A22B-16node-profile.sh.
#
# Usage:
#   bash prepare-qwen3-235b-A22B.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
MODEL_TYPE="qwen3-235B-A22B"
HF_REPO="Qwen/Qwen3-235B-A22B-Instruct-2507"
GPUS_PER_NODE=4

# ---------------------------------------------------------------------------
# Download model and datasets
# ---------------------------------------------------------------------------
mkdir -p /root/models /root/datasets
hf download "$HF_REPO" --local-dir "/root/models/${MODEL_NAME}"

python3 -c "
from miles.utils.external_utils.command_utils import hf_download_dataset
hf_download_dataset('zhuzilin/dapo-math-17k')
hf_download_dataset('zhuzilin/aime-2024')
"

# ---------------------------------------------------------------------------
# Convert checkpoint
# ---------------------------------------------------------------------------
mkdir -p /root/multinode
python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${GPUS_PER_NODE},
    dir_dst='/root/multinode',
)
"

echo "Prepare done."
