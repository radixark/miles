
#!/bin/bash

# Prepare script for Kimi-K2: download model, datasets, and convert checkpoint.
#
# Run this BEFORE run-kimi-k2-64node-profile.sh.
#
# NOTE: The rdma branch test uses Kimi-K2-Instruct-bf16 for the hf-checkpoint
# and conversion. If using the bf16 variant, change HF_REPO and MODEL_NAME below.
#
# Usage:
#   bash prepare-kimi-k2.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Kimi-K2-Instruct"
MODEL_TYPE="kimi-k2"
HF_REPO="moonshotai/Kimi-K2-Instruct"
GPUS_PER_NODE=8

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
# Convert checkpoint to megatron format
# (EP=8, decoder-last-pipeline-num-layers=5)
# ---------------------------------------------------------------------------
mkdir -p /root/multinode
python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${GPUS_PER_NODE},
    dir_dst='/root/multinode',
    extra_args='--expert-model-parallel-size 8 --decoder-last-pipeline-num-layers 5',
)
"

echo "Prepare done."
