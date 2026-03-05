#!/bin/bash

# SWE-Agent V2 Debug: Qwen3-4B with Harbor
# Minimal debug run to verify the full training pipeline

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Harbor server URL
export SWE_AGENT_URL="${SWE_AGENT_URL:-http://swe-env-maocheng:11000}"
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/root/harbor_tasks/swebench}"
# Docker containers spawned by Harbor need to reach Miles Router via Docker network
export MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-miles-maocheng}"

source /root/miles/scripts/models/qwen3-4B.sh

CKPT_ARGS=(
    --hf-checkpoint /root/Qwen3-4B
    --ref-load /root/Qwen3-4B_torch_dist
    --save /root/Qwen3-4B_miles_debug/
    --save-interval 100
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

ROLLOUT_ARGS=(
    --prompt-data /root/swe_train.jsonl
    --input-key prompt
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout 50
    --rollout-batch-size 4
    --n-samples-per-prompt 4
    --rollout-temperature 0.8
    --rollout-max-response-len 4096

    --global-batch-size 16
    --balance-data
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

WANDB_ARGS=()

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
    --use-miles-router
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# V2: Generic agentic generate + SWE-Agent custom agent function
CUSTOM_ARGS=(
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path swe_agent_function.run
    --custom-rm-path generate.reward_func
    --rollout-function-path generate.RolloutFn
    --dynamic-sampling-filter-path generate.dynamic_filter
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "Starting Ray cluster at ${MASTER_ADDR}..."
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=8899

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/miles\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"SWE_AGENT_URL\": \"${SWE_AGENT_URL}\",
    \"SWE_AGENT_MODEL_NAME\": \"model\",
    \"MILES_ROUTER_EXTERNAL_HOST\": \"${MILES_ROUTER_EXTERNAL_HOST:-}\",
    \"HARBOR_TASKS_DIR\": \"${HARBOR_TASKS_DIR}\"
  }
}"

echo "Launching debug training..."
echo "  SWE Agent URL: ${SWE_AGENT_URL}"
echo "  Harbor tasks:  ${HARBOR_TASKS_DIR}"
echo "  Model: Qwen3-4B"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 /root/miles/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}

echo "Debug training completed!"
