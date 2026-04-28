# Training ↔ Rollout precision alignment sanity script.
#
# Freezes weights (`--debug-skip-optimizer-step`) so log_prob_new ≡ log_prob_old
# is purely a function of the two DiT implementations' numerical floor
# (diffusers train side vs sglang-d rollout side). Use this to:
#   1) detect regressions in the bf16 floor after sglang / diffusers / FSDP
#      upgrades,
#   2) verify scheduler / CFG / timestep-scale / bf16-cast fixes still hold,
#   3) produce reference `[noise_pred align ...]` lines for comparison.
#
# Target metrics (see DIFFUSION_TRAIN_ROLLOUT_ALIGNMENT.md):
#   log_prob_mean_abs_diff   : 1e-4 ~ 3e-4 with ignore_last=2 (strict < 1e-3)
#                              3e-4 ~ 6e-4 with ignore_last=1 (flow_grpo default)
#   approx_kl                : ~ 0 (shows 0.0000 in .4f format)
#   log_prob_new == log_prob_old to 4+ decimals
#   noise_pred align mean    : ~ 2.8e-2 (bf16 floor between the two DiTs)
#
# Usage:
#   nohup bash scripts/run-diffusion-grpo-ocr-align.sh \
#     > logs/diffusion_align_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
#!/usr/bin/env bash

# for rerun the task
pkill -9 sgl*
sleep 3
ray stop --force
pkill -9 ray*
pkill -9 python*
sleep 3
pkill -9 ray*
pkill -9 python*


# pkill can't reap zombies — kill their live parents so init reaps them.
ps -eo ppid,state,comm --no-headers | awk '$2=="Z" && $1!=1 && $3~/ray|python|sglang/ {print $1}' | sort -u | xargs -r kill -9 2>/dev/null || true
sleep 2

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Align runs need only 2 GPUs (1 train rank + 1 rollout engine, colocated).
# Bump / change to match your free GPUs.
export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="diffusion_align_$(date +%Y%m%d_%H%M%S)"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-align
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
  )
fi
# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# Smallest rollout that still exercises all machinery (prompt → rollout →
# training forward → PPO loss + metrics). Not meant to train; meant to print
# `log_prob_mean_abs_diff`, `approx_kl`, and `[noise_pred align ...]`.
python -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 8 \
  --num-rollout 100000 \
  --diffusion-timestep-batch 10 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 2 \
  --colocate \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 128 \
  --diffusion-init-lora-weight gaussian \
  --use-miles-router \
  --sglang-server-concurrency 4 \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward ocr:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-std \
  --rm-type ocr \
  --fsdp-master-dtype fp32 \
  --diffusion-rollout-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-true-cfg-scale 4.0 \
  --diffusion-noise-level 1.2 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
  --diffusion-sde-window-size 2 \
  --diffusion-sde-window-range 0,5 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  --global-batch-size 8 \
  --diffusion-debug-mode \
  --debug-skip-optimizer-step \
  --eval-prompt-data ocr_test "${ROOT_DIR}/data/ocr/test.jsonl" \
  --eval-interval 50 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
