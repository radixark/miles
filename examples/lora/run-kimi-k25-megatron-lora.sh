#!/bin/bash

# Kimi-K2.5 LoRA GRPO — 16 nodes × 8 GPUs (H200), colocated.
# Inherits the full-param Kimi-K2.5 recipe and only overrides LoRA-specific
# bits (rank/alpha, target modules, shared-outer adapters, LR, parallelism).

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
source "${SCRIPT_DIR}/../../scripts/models/kimi-k2-thinking.sh"

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/Kimi-K2.5-int4
   --ref-load $BASE_DIR/Kimi-K2.5-bf16
   --megatron-to-hf-mode bridge
   --model-name kimi_k25
)

LORA_ARGS=(
   --lora-rank 32                       # LoRA rank (typical values: 8, 16, 32, 64)
   --lora-alpha 32                      # LoRA alpha (usually equal to rank for RL)
   --lora-dropout 0.0                   # LoRA dropout (0.0 for RL training)
   --target-modules "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"
   --experts-shared-outer-loras         # shared A on fc1 / shared B on fc2 across experts
   --lora-base-cpu-backup               # keep frozen base on CPU to free GPU
   --no-gradient-accumulation-fusion
   --sglang-lora-backend triton         # !!! must for moe-lora !!!
   --sglang-lora-use-virtual-experts    # virtual-experts MoE LoRA path
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler

   --num-rollout 20
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 256
   --filter-zero-reward-samples
   --use-dynamic-global-batch-size
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime $BASE_DIR/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 8
   --expert-model-parallel-size 64
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 30

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   # Off-policy IS correction: PPO operates on within-train ratio; TIS clamps
   # the cross-engine (sglang Marlin int4 vs Megatron fake-QAT bf16) ratio with
   # a wider bound than PPO's eps_clip, keeping kernel-rounding bias out of
   # PPO clipping.
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5                            # PEFT tolerates ~10x full-param LR
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --use-distributed-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-kimi-k25
   --wandb-group kimi-k25-lora
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-server-concurrency 1024
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
   --use-rollout-routing-replay
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-check-for-nan-in-loss-and-grad
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_TIMEOUT\": \"3600\",
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"32\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --use-miles-router \
   --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 )) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
