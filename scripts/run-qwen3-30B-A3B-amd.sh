#!/bin/bash

# for rerun the task
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python
# pkill -9 redis

set -euxo pipefail

# ==================== Platform Detection ====================
if [ -e /dev/kfd ] || python3 -c "import torch; assert torch.version.hip" 2>/dev/null; then
    GPU_VENDOR="amd"
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
else
    echo "ERROR: No supported GPU detected (need NVIDIA or AMD)"
    exit 1
fi
echo "Detected GPU vendor: ${GPU_VENDOR}"

# ==================== Configurable Paths ====================
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
export MODEL_DIR DATA_DIR

# ==================== Platform-Specific Setup ====================
if [ "$GPU_VENDOR" = "amd" ]; then
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    if [ -z "${HIP_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=0
    else
        NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    fi
    HAS_NVLINK=0
else
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
    if [ "$NVLINK_COUNT" -gt 0 ]; then
        HAS_NVLINK=1
    else
        HAS_NVLINK=0
    fi
    echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

# Checkpoint behavior:
# - By default, resume from the previous MILES checkpoint directory.
# - Set RESET_LOAD=1 to ignore stale/incompatible MILES checkpoints and start from torch_dist.
LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_miles/"
if [ "${RESET_LOAD:-0}" = "1" ]; then
   LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_torch_dist"
fi
LOAD_DIR="${LOAD_DIR:-${LOAD_DIR_DEFAULT}}"
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/Qwen3-30B-A3B_miles/}"

echo "LOAD_DIR=${LOAD_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B
   #--hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-FP8
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B_torch_dist
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --log-passrate

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

# AMD: disable custom all-reduce and use triton MoE backend
if [ "$GPU_VENDOR" = "amd" ]; then
    SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
   #  SGLANG_ARGS+=(--sglang-attention-backend triton)
   #  SGLANG_ARGS+=(--sglang-moe-runner-backend triton_kernel)
fi

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# ==================== Launch Ray ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8285

# Dynamically detect Megatron-LM installation path
MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

# Build the runtime environment JSON
AITER_ENV=""
if [ "$GPU_VENDOR" = "amd" ]; then
    # AITER enabled. Qwen3-30B-A3B has moe_intermediate_size=768. At TP=8,
    # inter_dim=768/8=96, which the aiter CK MoE GEMM (ck_moe_stage1) does not support
    # (requires inter_dim % 64 == 0). sglang.patch adds a guard in unquant.py to
    # detect this and fall back to triton automatically for unsupported inter_dim.
    AITER_ENV=",\"SGLANG_USE_AITER\": \"1\""
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_SET_CPU_AFFINITY\": \"0\"${AITER_ENV}
  }
}"
# Note: SGLANG_SET_CPU_AFFINITY=0 is required on AMD to avoid CPU affinity errors
# when the container exposes fewer CPUs than the full machine topology.

ray job submit --address="http://127.0.0.1:8285" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
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
