#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

SMOKE_COLOCATE="${SMOKE_COLOCATE:-1}"
SMOKE_ACTOR_GPUS_PER_NODE="${SMOKE_ACTOR_GPUS_PER_NODE:-2}"
SMOKE_ROLLOUT_GPUS="${SMOKE_ROLLOUT_GPUS:-2}"
SMOKE_ROLLOUT_GPUS_PER_ENGINE="${SMOKE_ROLLOUT_GPUS_PER_ENGINE:-1}"
SMOKE_PICKSCORE_NUM_WORKERS="${SMOKE_PICKSCORE_NUM_WORKERS:-1}"
SMOKE_PICKSCORE_NUM_GPUS_PER_WORKER="${SMOKE_PICKSCORE_NUM_GPUS_PER_WORKER:-1.0}"
SMOKE_PICKSCORE_BATCH_SIZE="${SMOKE_PICKSCORE_BATCH_SIZE:-2}"

COLOCATE_ARGS=()
if [[ "${SMOKE_COLOCATE}" == "1" || "${SMOKE_COLOCATE}" == "true" || "${SMOKE_COLOCATE}" == "yes" ]]; then
  # Use two colocated train/rollout GPUs plus one dedicated PickScore reward GPU.
  DEFAULT_CUDA_VISIBLE_DEVICES="4,5,6"
  DEFAULT_NUM_GPUS_PER_NODE="3"
  COLOCATE_ARGS+=(--colocate)
else
  # Use two train GPUs, two rollout GPUs, and one dedicated PickScore reward GPU.
  DEFAULT_CUDA_VISIBLE_DEVICES="1,2,3,4,5"
  DEFAULT_NUM_GPUS_PER_NODE="5"
fi

SMOKE_NUM_GPUS_PER_NODE="${SMOKE_NUM_GPUS_PER_NODE:-${DEFAULT_NUM_GPUS_PER_NODE}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="diffusion_grpo_pickscore_smoke_$(date +%Y%m%d_%H%M%S)"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 4
    --diffusion-log-image-interval 1
    --disable-wandb-random-suffix
  )
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 2 \
  --num-rollout 1 \
  --diffusion-timestep-batch 10 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node "${SMOKE_ACTOR_GPUS_PER_NODE}" \
  --rollout-num-gpus "${SMOKE_ROLLOUT_GPUS}" \
  --rollout-num-gpus-per-engine "${SMOKE_ROLLOUT_GPUS_PER_ENGINE}" \
  --num-gpus-per-node "${SMOKE_NUM_GPUS_PER_NODE}" \
  "${COLOCATE_ARGS[@]}" \
  --no-offload-rollout \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --diffusion-init-lora-weight gaussian \
  --use-miles-router \
  --sglang-server-concurrency 2 \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type pickscore \
  --pickscore-num-workers "${SMOKE_PICKSCORE_NUM_WORKERS}" \
  --pickscore-num-gpus-per-worker "${SMOKE_PICKSCORE_NUM_GPUS_PER_WORKER}" \
  --pickscore-batch-size "${SMOKE_PICKSCORE_BATCH_SIZE}" \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --fsdp-master-dtype fp32 \
  --diffusion-rollout-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-true-cfg-scale 4.0 \
  --diffusion-rollout-noise-level 1.2 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
  --diffusion-sde-window-size 2 \
  --diffusion-sde-window-range 0,5 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  --global-batch-size 2 \
  --diffusion-ignore-last 1 \
  --diffusion-rollout-debug-mode \
  --debug-skip-optimizer-step \
  --eval-prompt-data ocr_test "${ROOT_DIR}/data/ocr/test.jsonl" \
  --eval-interval 1 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
