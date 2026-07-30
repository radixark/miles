#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

TINKER_MODEL_PATH="${TINKER_MODEL_PATH:-/root/Qwen3-0.6B}"
TINKER_MODEL_NAME="${TINKER_MODEL_NAME:-Qwen/Qwen3-0.6B}"
TINKER_CHECKPOINT_DIR="${TINKER_CHECKPOINT_DIR:-/root/tinker-checkpoints}"
TINKER_API_PORT="${TINKER_API_PORT:-8068}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
TINKER_MAX_MODELS="${TINKER_MAX_MODELS:-8}"
TINKER_TRAIN_GPUS="${TINKER_TRAIN_GPUS:-1}"
TINKER_ROLLOUT_GPUS="${TINKER_ROLLOUT_GPUS:-1}"
TINKER_TP_SIZE="${TINKER_TP_SIZE:-1}"

RUNTIME_ENV_JSON="$(
  TINKER_RUNTIME_REPO_ROOT="${REPO_ROOT}" \
  TINKER_RUNTIME_MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" \
    python3 -c '
import json
import os

env_vars = {
    "PYTHONPATH": (
        os.environ["TINKER_RUNTIME_MEGATRON_LM_PATH"]
        + ":"
        + os.environ["TINKER_RUNTIME_REPO_ROOT"]
    ),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUNBUFFERED": "1",
}
if api_key := os.environ.get("TINKER_API_KEY"):
    env_vars["TINKER_API_KEY"] = api_key
print(json.dumps({"env_vars": env_vars}))
'
)"

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${REPO_ROOT}/train_tinker.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TINKER_TRAIN_GPUS}" \
  --rollout-num-gpus "${TINKER_ROLLOUT_GPUS}" \
  --use-miles-router \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${TINKER_MODEL_PATH}" \
  --megatron-to-hf-mode bridge \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head \
  --multi-lora-n-adapters "${TINKER_MAX_MODELS}" \
  --multi-lora-idle-poll-s 1 \
  --multi-lora-max-coalesce-wait-s 0.5 \
  --multi-lora-api-port "${TINKER_API_PORT}" \
  --tinker-model-name "${TINKER_MODEL_NAME}" \
  --tinker-tokenizer-id "${TINKER_MODEL_PATH}" \
  --tinker-checkpoint-dir "${TINKER_CHECKPOINT_DIR}" \
  --pause-generation-mode in_place \
  --max-weight-staleness 3 \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 64 \
  --rollout-temperature 1 \
  --global-batch-size 1 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-4 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --tensor-model-parallel-size "${TINKER_TP_SIZE}" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 4096 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash
