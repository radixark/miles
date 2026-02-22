#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

BASE_FOLDER=/root/shared
# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/small-next.sh"

CKPT_ARGS=(
   # --hf-checkpoint ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking
   # --ref-load ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_torch_dist
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking-8L
   --ref-load ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_partial_torch_dist
   # --load ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_miles/
   # --save ${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_miles/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 2
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 16
   --balance-data
)

EVAL_ARGS=(
   # --eval-interval 20
   --eval-prompt-data aime ${BASE_FOLDER}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --qkv-format bshd
   # --qkv-format thd
   # --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator gspo
   #--use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen3-next-80B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   # --sglang-ep-size 8

   # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 64)

   # mtp
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 2
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 3

   # --sglang-max-running-requests 64

   # use triton backend to avoid flashinfer_trtllm block layout incompatibility
   # with update_weights_from_tensor (process_weights_after_loading converts weights
   # to 3D block layout but weight updates send 2D HF-format tensors)
   # --sglang-moe-runner-backend triton
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --check-weight-update-equal
   # --debug-first-weight-sync ${BASE_FOLDER}/debug-first-weight-sync/

   # --moe-token-dispatcher-type flex
   # --moe-enable-deepep
   # --debug-rollout-only
   # --save-debug-rollout-data /root/shared/qwen-next/debug1/data_{rollout_id}.pt
   # --debug-train-only
   # --load-debug-rollout-data /root/shared/qwen-next/debug1/data_{rollout_id}.pt
)

SPEC_ARGS=(
   # --enable-mtp-training
   # --mtp-loss-scaling-factor 0.2
)

# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# # Build the runtime environment JSON with proper variable substitution
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
#   }
# }"

# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json="${RUNTIME_ENV_JSON}" \
#    -- python3 train.py \
#    --actor-num-nodes 1 \
#    --actor-num-gpus-per-node 4 \
#    --rollout-num-gpus 2 \
#    ${MODEL_ARGS[@]} \
#    ${CKPT_ARGS[@]} \
#    ${ROLLOUT_ARGS[@]} \
#    ${OPTIMIZER_ARGS[@]} \
#    ${GRPO_ARGS[@]} \
#    ${WANDB_ARGS[@]} \
#    ${PERF_ARGS[@]} \
#    ${EVAL_ARGS[@]} \
#    ${SGLANG_ARGS[@]} \
#    ${MISC_ARGS[@]} \
#    ${SPEC_ARGS[@]}

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
   ${SPEC_ARGS[@]}

# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json="${RUNTIME_ENV_JSON}" \
#    -- python3 train.py \
#    --actor-num-nodes 1 \
#    --actor-num-gpus-per-node 8 \
#    --colocate \
#    ${MODEL_ARGS[@]} \
#    ${CKPT_ARGS[@]} \
#    ${ROLLOUT_ARGS[@]} \
#    ${OPTIMIZER_ARGS[@]} \
#    ${GRPO_ARGS[@]} \
#    ${WANDB_ARGS[@]} \
#    ${PERF_ARGS[@]} \
#    ${EVAL_ARGS[@]} \
#    ${SGLANG_ARGS[@]} \
#    ${MISC_ARGS[@]} \
#    ${SPEC_ARGS[@]}
