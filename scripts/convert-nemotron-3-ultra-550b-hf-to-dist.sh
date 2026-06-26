#!/bin/bash
# Offline HF -> Megatron torch_dist conversion for Nemotron-3-Ultra-550B-A55B.
# Run once; the RL launcher then uses --load <dist> instead of paying the slow
# 512-expert HF bridge mapping on every start.
#
# The conversion layout MUST match the RL run's layout (no reshard at load):
#   TP8  PP4  EP32  ETP1  => 128 ranks / 16 nodes (8 GPU each).
#
# Usage on each pod (run on all 16, ranks 0..15; rank 0 == head):
#   bash convert-nemotron-3-ultra-550b-hf-to-dist.sh <NODE_RANK 0..15> <HEAD_IP>

set -u
NODE_RANK=${1:?Usage: $0 <node_rank 0..15> <head_ip>}
HEAD_IP=${2:?Usage: $0 <node_rank 0..15> <head_ip>}
NNODES=${NNODES:-16}
PORT=${PORT:-29531}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/nemotron-3-ultra-550b-a55b.sh"   # provides MODEL_ARGS

MODELS_DIR=${MODELS_DIR:-/cluster_public/miles_data/models}
# Public HF repo (https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).
# Override HF=<local path> to use a pre-downloaded copy on shared storage.
HF=${HF:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
SAVE=${SAVE:-$MODELS_DIR/nemotron-3-ultra-550b-a55b_torch_dist}

cd "${SCRIPT_DIR}/.."
export PYTHONPATH=/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$SAVE"

torchrun --nnodes="$NNODES" --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$HEAD_IP" --master-port="$PORT" \
  tools/convert_hf_to_torch_dist_bridge.py \
  "${MODEL_ARGS[@]}" \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --expert-model-parallel-size 32 \
  --expert-tensor-parallel-size 1 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --hf-checkpoint "$HF" \
  --save "$SAVE" \
  --megatron-to-hf-mode bridge
