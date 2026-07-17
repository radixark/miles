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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export HF_HOME=/workspace/hf_cache

# Load model architecture config
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gpt-oss-20b.sh"

BASE_DIR=/root/shared

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/gpt-oss-20b
   # --hf-checkpoint $BASE_DIR/gpt-oss-20b-BF16
   --megatron-to-hf-mode bridge
   # --save $BASE_DIR/gpt-oss-20b-BF16
   # --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 1000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   # Parallelism: TP=8, EP=4, PP=1, CP=1
   # SP is required when combining TP + EP
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   # Recomputation: full recompute needed to fit optimizer states in 80GB
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # Batch size settings
   # Note: --use-dynamic-batch-size is not supported with --qkv-format bshd
   --micro-batch-size 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # TODO: need gpt oss ckpt conversion.
   # --use-kl-loss
   # --kl-loss-coef 0.00
   # --kl-loss-type low_var_kl
   # --kl-coef 0.00
   --entropy-coef 0.00
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
   # CPU offload optimizer states (fp32 master weights + Adam moments) to free ~30GB GPU memory
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   # TP size for sglang inference engine
   --rollout-num-gpus-per-engine 4
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
   --sglang-mem-fraction-static 0.70
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-gpt-oss
   # --wandb-group "20b-bf16"
   # --wandb-key ${WANDB_API_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # Sink attention (sliding window + learnable softmax) in TE only supports BSHD/SBHD, not THD.
   # Must use --qkv-format bshd for the fused backend to work with this model's attention pattern.
   --qkv-format bshd
   --attention-backend fused
)


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
   --actor-num-gpus-per-node 8 \
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
   ${MISC_ARGS[@]}
