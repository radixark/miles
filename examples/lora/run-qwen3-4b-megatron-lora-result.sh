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

# export SGLANG_LORA_PROFILE=1
# export SGLANG_LORA_PROFILE_INTERVAL=10
# export SGLANG_LORA_ENABLE_FUSION=1

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

LR=2e-5

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /root/miles/scripts/models/qwen3-4B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --save /root/Qwen3-4B-lora-ckpt
   --save-interval 50
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   --lora-dropout 0.0  # +fsdp
   --target-modules all-linear
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100 
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime24 /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-team miles-lora
   --wandb-project miles-lora-test
   --wandb-group qwen3-4B-megatron-lora-dapo-lr${LR}
   --disable-wandb-random-suffix
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-decode-log-interval 1000
   # --sglang-enable-metrics # -fsdp
   --sglang-mem-fraction-static 0.4 # +fsdp, memory usage on H200 = 140*0.4=56GB per GPU
   # --sglang-attention-backend fa3  # +fsdp
   # --sglang-attention-backend flashinfer
   --sglang-chunked-prefill-size 4096
)

MEGATRON_ARGS=(
   # --no-offload-train
   # --no-offload-rollout
   --megatron-to-hf-mode bridge
   # --offload-rollout-level kv_cache weight  # -fsdp: not supported in megatron
   # --train-backend fsdp  # -fsdp: use megatron instead
   --train-backend megatron  # +fsdp
   --attention-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --hidden-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --accumulate-allreduce-grads-in-fp32  # +fsdp, megatron specific
   --attention-softmax-in-fp32  # +fsdp, megatron specific
   # --attention-backend flash  # +fsdp, megatron specific
   # --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' # +fsdp, otherwise OOM
)

PERF_ARGS=(
   # --gradient-checkpointing # +fsdp
   # --sequence-parallel # +fsdp
   # --use-dynamic-batch-size # +fsdpF
   # --max-tokens-per-gpu 9216 # +fsdp, perf
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${NUM_GPUS}
   --colocate
   --calculate-per-token-loss # +fsdp
   --use-miles-router # +fsdp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats


RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
  }
}'


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MEGATRON_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"