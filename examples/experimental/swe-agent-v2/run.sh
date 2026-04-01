#!/bin/bash
# Agent V2 launcher: Miles <-> Harbor agent orchestration.
#
# Supports any task type (SWE-bench, Terminal-Bench, custom) via Harbor.

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILES_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$MILES_ROOT/scripts/models/glm4.7-flash.sh"

BASE_DIR=/root/shared
AGENT_SERVER_URL="${AGENT_SERVER_URL:-${SWE_AGENT_URL:-http://agent_env:11000}}"
HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/root/harbor_tasks}"
ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-$(hostname)}"

CKPT_ARGS=(
  --hf-checkpoint $BASE_DIR/GLM-4.7-Flash
  --ref-load $BASE_DIR/GLM-4.7-Flash_torch_dist
  --save $BASE_DIR/GLM-4.7-Flash_agent_v2/
  --save-interval 100
)

ROLLOUT_ARGS=(
  --prompt-data /root/swe_train.jsonl
  --input-key prompt
  --metadata-key metadata
  --rollout-shuffle

  --num-rollout 3000
  --rollout-batch-size 8
  --n-samples-per-prompt 8
  --rollout-temperature 0.8
  --rollout-max-response-len 8192
  --global-batch-size 64
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  # --sglang-speculative-algorithm EAGLE
  # --sglang-speculative-num-steps 2
  # --sglang-speculative-eagle-topk 1
  # --sglang-speculative-num-draft-tokens 3
  --sglang-mem-fraction-static 0.7
  --sglang-tool-call-parser glm47
  --sglang-reasoning-parser glm45

  --use-miles-router
  --sglang-router-port 30000
)

AGENT_ARGS=(
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path swe_agent_function.run
  --custom-rm-path generate.reward_func
  --rollout-function-path generate.RolloutFn
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
  --tito-model glm47
  --chat-template-path autofix
  --use-session-server
)

WANDB_ARGS=(
  # --use-wandb
  # --wandb-project miles-agent-v2
  # --wandb-group agent-v2
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

DEBUG_ARGS=(
  --debug-rollout-only
)

# ── Start Ray ────────────────────────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus 8 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --port=8899

RUNTIME_ENV=$(python3 -c "
import json, sys
print(json.dumps({'env_vars': {
    'PYTHONPATH': '/root/Megatron-LM/:${SCRIPT_DIR}:${MILES_ROOT}',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'MILES_EXPERIMENTAL_ROLLOUT_REFACTOR': '1',
    'AGENT_SERVER_URL': '${AGENT_SERVER_URL}',
    'AGENT_MODEL_NAME': '${AGENT_MODEL_NAME:-model}',
    'MILES_ROUTER_EXTERNAL_HOST': '${ROUTER_EXTERNAL_HOST}',
    'HARBOR_TASKS_DIR': '${HARBOR_TASKS_DIR}',
    'MILES_HOST_IP': '${MILES_HOST_IP:-$(hostname)}',
    'NCCL_NVLS_ENABLE': '0',
}}))
")

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV" \
  -- python3 "$MILES_ROOT/train.py" \
  --colocate \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${AGENT_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${DEBUG_ARGS[@]}"
