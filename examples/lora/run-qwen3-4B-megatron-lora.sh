#!/bin/bash

# Example launcher that reuses the Qwen3-4B recipe but delegates evaluation to an
# external Nemo Skills server via the eval_delegate_rollout wrapper.

# Clean up any stale processes from a previous run.
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

SKILLS_OPENAI_MODEL_NAME=${SKILLS_OPENAI_MODEL_NAME:-"miles-openai-model"}

export GPUS_PER_NODE=4
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/miles/scripts/models/qwen3-4B.sh"

# Store eval/delegate settings in a YAML config similar to examples/eval_multi_task.
# EVAL_CONFIG_PATH=${SKILLS_EVAL_CONFIG_PATH:-"${REPO_ROOT}/examples/eval/scripts/multi_tasks.yaml"}
EVAL_CONFIG_PATH=${SKILLS_EVAL_CONFIG_PATH:-"${REPO_ROOT}/miles/examples/eval/scripts/multi_tasks.yaml"}


CKPT_ARGS=(
   # --hf-checkpoint /root/Qwen3-4B
   --hf-checkpoint /root/models/Qwen3-4B
   --megatron-to-hf-mode bridge
)


LORA_ARGS=(
   --lora-rank 32                    # LoRA rank (typical values: 8, 16, 32, 64)
   --lora-alpha 32                   # LoRA alpha (usually 2x rank)
   --lora-dropout 0.0                # LoRA dropout (0.0 for RL training)
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)



ROLLOUT_ARGS=(
   # --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   # --rollout-batch-size 32
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   # --rollout-max-response-len 8192
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --over-sampling-batch-size 64

   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   # --global-batch-size 256
   --global-batch-size 128
   --balance-data
)

# EVAL_ARGS=(
#    --eval-interval 5
#    --eval-config "${EVAL_CONFIG_PATH}"
#    --eval-function-path examples.eval.eval_delegate_rollout.generate_rollout
# )

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 16384
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # --lr 1e-6
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-team miles-lora
   --wandb-project miles-lora-megatron
   --wandb-group qwen3-4B-test
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # --sglang-mem-fraction-static 0.4

   # --sglang-enable-deterministic-inference
   # --sglang-attention-backend flashinfer
   # --deterministic-mode
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export CUDA_VISIBLE_DEVICES=0,1
# Set Up Your GPUs for Training

# export GPUS_PER_NODE=2 #default


ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPUS_PER_NODE --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


# ray job submit --address="http://127.0.0.1:8265" \
   # --runtime-env-json="${RUNTIME_ENV_JSON}" \

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
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
   ${LORA_ARGS[@]} 
