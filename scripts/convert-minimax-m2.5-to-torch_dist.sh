#!/usr/bin/env bash
# MiniMax-M2.5: HF checkpoint -> Megatron distributed (torch_dist) checkpoint.
#
# Only the *launch* path differs from a plain ``python tools/convert_hf_to_torch_dist.py``:
# use ``torchrun`` so each rank builds/shards the model and ``bridge.load_weights`` runs
# under the same parallel layout.
#
# Prerequisites (inside miles docker image, e.g. miles_ycy):
#   - Miles repo and HF weights mounted at the paths below.
#   - Enough GPUs: MiniMax-M2.5 MoE does not fit on a single 80GB card at TP=1 EP=1.
#   - TP/CP > 1: Megatron requires CUDA_DEVICE_MAX_CONNECTIONS=1 (set below by default).
#
# IMPORTANT (tools/convert_hf_to_torch_dist.py get_args):
#   When WORLD_SIZE > 1 and ``--pipeline-model-parallel-size`` is still 1, the script
#   *forces* pipeline parallel to WORLD_SIZE (pure layer slicing). MoE experts stay
#   fully replicated on each pipeline stage, which often still OOMs.
#   So for multi-GPU conversion, pass an explicit PP (and TP/EP) with PP != 1 and
#   ensure TP × PP × EP × CP × … == nproc_per_node (see Megatron constraints).
#
# Usage:
#   cd /home/yangchengyi/data/miles
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/convert-minimax-m2.5-to-torch_dist.sh
#
# Tunables via env:
#   MILES_ROOT, HF_CKPT, TORCH_DIST_OUT, NPROC_PER_NODE, MASTER_PORT,
#   TENSOR_MODEL_PARALLEL_SIZE, PIPELINE_MODEL_PARALLEL_SIZE,
#   EXPERT_MODEL_PARALLEL_SIZE, CONTEXT_PARALLEL_SIZE, MINIMAX_PADDED_VOCAB_SIZE,
#   PYTORCH_CUDA_ALLOC_CONF

set -euo pipefail

MILES_ROOT="${MILES_ROOT:-/home/yangchengyi/data/miles}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
export PYTHONPATH="${MILES_ROOT}:${MEGATRON_ROOT}"

# Required by Megatron when tensor-model-parallel-size > 1 or context-parallel-size > 1
# (assert in megatron/training/arguments.py validate_args).
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Reduce allocator fragmentation during large module construction.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_name="${MODEL_NAME:-MiniMax-M2.5}"
HF_CKPT="${HF_CKPT:-/home/yangchengyi/data/models/${model_name}}"
TORCH_DIST_OUT="${TORCH_DIST_OUT:-/home/yangchengyi/data/models/${model_name}_torch_dist}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29526}"

# Default 8-GPU slice (OOM-safer for MiniMax-M2.5 MoE):
# TP=1 × PP=2 × EP=4 = 8. This lowers local experts per rank (256/EP) from 128 -> 64.
# If you have 4 GPUs: NPROC_PER_NODE=4 TP=1 PP=2 EP=2.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-2}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-4}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

cd "${MILES_ROOT}"
# shellcheck source=/dev/null
source "${MILES_ROOT}/scripts/models/minimax-m2.5.sh"

CONVERT_PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
)

# HF ``embed_tokens`` is exactly vocab_size=200064; Megatron's default padding with TP>1
# can raise padded_vocab_size (e.g. 200192), then mbridge scatter expects (100096,.*) per
# rank while HF shards are (100032,.*) — set explicit padding to match the checkpoint.
CONVERT_EXTRA_ARGS=(
  --padded-vocab-size "${MINIMAX_PADDED_VOCAB_SIZE:-200064}"
)

exec torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${MILES_ROOT}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  "${CONVERT_PARALLEL_ARGS[@]}" \
  "${CONVERT_EXTRA_ARGS[@]}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${TORCH_DIST_OUT}"
