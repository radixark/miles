#!/bin/bash

# Prepare script for GLM-4.7-Flash: install dependencies, download model/datasets
# and convert checkpoint.
#
# Run this BEFORE run-glm4.7-flash-2node-profile.sh.
#
# Usage:
#   bash prepare-glm4.7-flash.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="GLM-4.7-Flash"
MODEL_TYPE="glm4.7-flash"
HF_REPO="zai-org/GLM-4.7-Flash"
GPUS_PER_NODE=8

# ---------------------------------------------------------------------------
# Install transformers version that supports GLM-4.7-Flash
# ---------------------------------------------------------------------------
pip install git+https://github.com/huggingface/transformers.git@76732b4e7120808ff989edbd16401f61fa6a0afa

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
