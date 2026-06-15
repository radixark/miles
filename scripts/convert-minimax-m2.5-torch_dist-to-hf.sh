#!/usr/bin/env bash
# MiniMax-M2.5: Megatron distributed (torch_dist) checkpoint -> HF checkpoint.

set -euo pipefail

MILES_ROOT="${MILES_ROOT:-/home/yangchengyi/data/miles}"
model_name="${MODEL_NAME:-MiniMax-M2.5}"
HF_CKPT="${HF_CKPT:-/home/yangchengyi/data/models/${model_name}}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/home/yangchengyi/data/models/${model_name}_miles}"
INPUT_DIR="${INPUT_DIR:-${MEGATRON_CKPT}/release}"
SAVE_DIR="${SAVE_DIR:-/home/yangchengyi/data/models/${model_name}_hf_output}"

cd "${MILES_ROOT}"

exec python tools/convert_torch_dist_to_hf.py \
  --model-name minimax_m2 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${SAVE_DIR}" \
  --origin-hf-dir "${HF_CKPT}" \
  --vocab-size 200064
