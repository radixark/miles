#!/bin/bash

# Unified Training Script - extends run-qwen3-4B.sh with conversion stages
# Usage: ./unified_train.sh [--skip-convert-to-megatron] [--skip-training] [--skip-convert-to-hf]

# Parse flags
SKIP_CONVERT_TO_MEGATRON=false
SKIP_TRAINING=false
SKIP_CONVERT_TO_HF=false
OVERWRITE_MEGATRON=false

for arg in "$@"; do
    case $arg in
        --skip-convert-to-megatron) SKIP_CONVERT_TO_MEGATRON=true ;;
        --skip-training) SKIP_TRAINING=true ;;
        --skip-convert-to-hf) SKIP_CONVERT_TO_HF=true ;;
        --overwrite-megatron) OVERWRITE_MEGATRON=true ;;
    esac
done

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

#############################
# Configuration - EDIT THESE
#############################
HF_CHECKPOINT=/root/Qwen3-4B
TORCH_DIST_DIR=/root/Qwen3-4B_torch_dist
TRAINING_DIR=/root/Qwen3-4B_miles
HF_OUTPUT_DIR=/root/Qwen3-4B_output_hf
MEGATRON_LM_PATH=/root/Megatron-LM

#############################
# Stage 1: HuggingFace -> Megatron
#############################
if [ "$SKIP_CONVERT_TO_MEGATRON" = false ]; then
    echo "=== Stage 1: Converting HuggingFace to Megatron ==="
    cd "$SCRIPT_DIR/.."
    
    if [ -d "$TORCH_DIST_DIR/release" ]; then
        if [ "$OVERWRITE_MEGATRON" = true ]; then
            echo "Megatron checkpoint exists. Overwriting as requested..."
            rm -rf "$TORCH_DIST_DIR"
            
            PYTHONPATH="$MEGATRON_LM_PATH" python tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint "$HF_CHECKPOINT" \
                --save "$TORCH_DIST_DIR"
        else
            echo "Megatron checkpoint already exists at $TORCH_DIST_DIR/release. Skipping conversion."
            echo "Use --overwrite-megatron to force regeneration."
        fi
    else
        PYTHONPATH="$MEGATRON_LM_PATH" python tools/convert_hf_to_torch_dist.py \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint "$HF_CHECKPOINT" \
            --save "$TORCH_DIST_DIR"
    fi
fi

#############################
# Stage 2: Training (original run-qwen3-4B.sh logic)
#############################
if [ "$SKIP_TRAINING" = false ]; then
    echo "=== Stage 2: Training ==="
    
    # for rerun the task
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 python || true
    sleep 3

    export PYTHONBUFFERED=16

    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
    if [ "$NVLINK_COUNT" -gt 0 ]; then
        HAS_NVLINK=1
    else
        HAS_NVLINK=0
    fi
    echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

    CKPT_ARGS=(
       --hf-checkpoint "$HF_CHECKPOINT"
       --ref-load "$TORCH_DIST_DIR"
       --load "$TRAINING_DIR"
       --save "$TRAINING_DIR"
       --save-interval 5
    )

    ROLLOUT_ARGS=(
       --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
       --input-key prompt
       --label-key label
       --apply-chat-template
       --rollout-shuffle
       --rm-type deepscaler
       --num-rollout 3000
       --rollout-batch-size 32
       --n-samples-per-prompt 8
       --rollout-max-response-len 8192
       --rollout-temperature 0.8
       --global-batch-size 256
       --balance-data
    )

    EVAL_ARGS=(
       --eval-interval 20
       --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
       --n-samples-per-eval-prompt 16
       --eval-max-response-len 16384
       --eval-top-p 0.7
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
    )

    WANDB_ARGS=(
       # --use-wandb
       # --wandb-project miles-dev
       # --wandb-group qwen3-4B-test
       # --wandb-key ${WANDB_KEY}
    )

    SGLANG_ARGS=(
       --rollout-num-gpus-per-engine 2
       --sglang-mem-fraction-static 0.7
    )

    MISC_ARGS=(
       --attention-dropout 0.0
       --hidden-dropout 0.0
       --accumulate-allreduce-grads-in-fp32
       --attention-softmax-in-fp32
       --attention-backend flash
    )

    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    RUNTIME_ENV_JSON="{
      \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
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
fi

#############################
# Stage 3: Megatron -> HuggingFace
#############################
if [ "$SKIP_CONVERT_TO_HF" = false ]; then
    echo "=== Stage 3: Converting Megatron to HuggingFace ==="
    cd "$SCRIPT_DIR/.."
    mkdir -p "$HF_OUTPUT_DIR"
    
    # Find all checkpoints (iter_* and release)
    CHECKPOINTS=$(find "$TRAINING_DIR" -maxdepth 1 -type d -name "iter_*" | sort -V)
    if [ -d "$TRAINING_DIR/release" ]; then
        CHECKPOINTS="$TRAINING_DIR/release"$'\n'"$CHECKPOINTS"
    fi
    
    if [ -n "$CHECKPOINTS" ]; then
        echo "$CHECKPOINTS" | while read -r CKPT; do
            [ -z "$CKPT" ] && continue
            CKPT_NAME=$(basename "$CKPT")
            
            if [ -d "${HF_OUTPUT_DIR}/${CKPT_NAME}" ]; then
                echo "Skipping $CKPT_NAME (already exists)"
                continue
            fi
            
            echo "Converting $CKPT_NAME..."
            PYTHONPATH="$MEGATRON_LM_PATH" python tools/convert_torch_dist_to_hf.py \
                --input-dir "$CKPT" \
                --output-dir "${HF_OUTPUT_DIR}/${CKPT_NAME}" \
                --origin-hf-dir "$HF_CHECKPOINT" \
                --force
            
            echo "Converted to: ${HF_OUTPUT_DIR}/${CKPT_NAME}"
        done
    else
        echo "No checkpoint found to convert"
    fi
fi

echo "=== Pipeline Complete ==="
