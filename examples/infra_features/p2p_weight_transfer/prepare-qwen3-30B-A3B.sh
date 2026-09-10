#!/bin/bash

# Prepare script for Qwen3-30B-A3B: download model/datasets and convert checkpoint.
#
# Run this BEFORE run-qwen3-30B-A3B-4node-profile.sh.
#
# Usage:
#   bash prepare-qwen3-30B-A3B.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen3-30B-A3B"
HF_REPO="Qwen/Qwen3-30B-A3B"
MODEL_TYPE="qwen3-30B-A3B"
GPUS_PER_NODE=4

# ---------------------------------------------------------------------------
# Download model and datasets
# ---------------------------------------------------------------------------
mkdir -p /root/models /root/datasets
hf download "$HF_REPO" --local-dir "/root/models/${MODEL_NAME}"

python3 -c "
from miles.utils.external_utils import command_utils
U = command_utils.default_config().create_backend()
U.hf_download_dataset('zhuzilin/dapo-math-17k')
U.hf_download_dataset('zhuzilin/aime-2024')
"

# ---------------------------------------------------------------------------
# Convert checkpoint
# ---------------------------------------------------------------------------
mkdir -p /root/multinode
python3 -c "
from miles.utils.external_utils import command_utils
U = command_utils.default_config().create_backend()
U.convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${GPUS_PER_NODE},
    dir_dst='/root/multinode',
)
"

echo "Prepare done."
