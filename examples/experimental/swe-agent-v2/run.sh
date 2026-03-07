#!/bin/bash

# Agent V2: Miles <-> Harbor agent orchestration with OpenAI chat format
# Supports any task type (SWE-bench, Terminal-Bench, custom) via Harbor

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Agent server URL (Harbor Trial API)
export AGENT_SERVER_URL="${AGENT_SERVER_URL:-${SWE_AGENT_URL:-http://swe_env:11000}}"
# Harbor task directories (created by adapters or prepare_harbor_tasks.py)
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/root/harbor_tasks}"

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
    --hf-checkpoint Qwen/Qwen3-4B-Instruct-2507
    --ref-load /root/qwen3-4B-Instruct-2507_torch_dist
    # --load /path/to/checkpoint/
    --save /root/qwen3-4B-Instruct-2507_miles_v2/
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

    # --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
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

EVAL_ARGS=(
    # --eval-interval 50
    # --eval-prompt-data /workspace/data/swe_gym_val.jsonl
    # --eval-input-key prompt
    # --eval-metadata-key metadata
    # --n-samples-per-eval-prompt 1
    # --eval-max-response-len 4096
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
if [ -n "$WANDB_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-miles-agent-v2}"
        --wandb-group "${WANDB_GROUP:-agent-v2-qwen3-4b}"
        --wandb-key "${WANDB_KEY}"
    )
    if [ -n "$WANDB_TEAM" ]; then
        WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
    fi
fi

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

# V2: Generic agentic generate + custom agent function
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
    \"AGENT_SERVER_URL\": \"${AGENT_SERVER_URL}\",
    \"AGENT_MODEL_NAME\": \"model\",
    \"MILES_ROUTER_EXTERNAL_HOST\": \"${MILES_ROUTER_EXTERNAL_HOST:-}\",
    \"HARBOR_TASKS_DIR\": \"${HARBOR_TASKS_DIR}\"
  }
}"

echo "Launching training..."
echo "  Agent server:  ${AGENT_SERVER_URL}"
echo "  Harbor tasks:  ${HARBOR_TASKS_DIR}"

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
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}

echo "Training completed!"
