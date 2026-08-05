#!/bin/bash

# Multi-node (8-node) disaggregated profiling script for GLM-4.5-Air (106B-A12B)
# with broadcast/p2p weight transfer.
#
# Configuration (from test_weight_transfer_moe_multinode_glm45_air_8nodes.py):
#   8 nodes total = 4 train + 4 rollout
#   Train:   TP=1, PP=4, CP=1, EP=8, ETP=1 (32 GPUs)
#   Rollout: 4 engines × 8 GPUs each (EP=8, DP attention)
#
# Usage:
#   bash run-glm4.5-air-8node-profile.sh <MODE> <NODE_RANK> <HEAD_NODE_IP>
#
#   MODE          : broadcast | p2p
#   NODE_RANK     : 0 (head node) | 1..7 (worker nodes)
#   HEAD_NODE_IP  : IP address of the head node
#
# Examples:
#   bash run-glm4.5-air-8node-profile.sh p2p 0 10.0.0.1   # head node
#   bash run-glm4.5-air-8node-profile.sh p2p 1 10.0.0.1   # worker node

set -ex

export PYTHONBUFFERED=16

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ $# -lt 3 ]; then
    echo "Usage: $0 <MODE> <NODE_RANK> <HEAD_NODE_IP>"
    echo "  MODE         : broadcast | p2p"
    echo "  NODE_RANK    : 0 (head) | 1..7 (workers)"
    echo "  HEAD_NODE_IP : IP of the head node"
    exit 1
fi

MODE="$1"              # broadcast | p2p
NODE_RANK="$2"         # 0 = head, 1..N = worker
HEAD_NODE_IP="$3"      # head node IP address

# ---------------------------------------------------------------------------
# Cleanup stale processes (ALL nodes in container env to avoid conflicts)
# ---------------------------------------------------------------------------
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
pkill -9 redis || true

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
NNODES=8
GPUS_PER_NODE=8
NUM_TRAIN_GPUS=32     # 4 nodes
NUM_ROLLOUT_GPUS=32   # 4 nodes
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
BUCKET_SIZE_GB="${BUCKET_SIZE_GB:-1.0}"
NO_SAVE_OPTIM=0
ENABLE_NCCL_NVLS=0

NUM_TRAIN_NODES=$((NUM_TRAIN_GPUS / GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="GLM-4.5-Air"
MODEL_TYPE="glm4.5-106B-A12B"

MILES_ROOT="/root/miles"
source "${MILES_ROOT}/scripts/models/${MODEL_TYPE}.sh"

# Rotary base override
export MODEL_ARGS_ROTARY_BASE=1000000

echo ""
echo "============================================================"
echo "  Model      : ${MODEL_NAME} (${MODEL_TYPE})"
echo "  Mode       : ${MODE}"
echo "  Nodes      : ${NNODES} total (${NUM_TRAIN_NODES} train + $((NNODES - NUM_TRAIN_NODES)) rollout)"
echo "  Parallelism: TP=1 PP=4 CP=1 EP=8"
echo "  Node rank  : ${NODE_RANK}, Head: ${HEAD_NODE_IP}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Execute one mode (broadcast or p2p)
# ---------------------------------------------------------------------------
run_mode() {
    local mode="$1"

    # --- Checkpoint ---
    # CKPT_SAVE_DIR allows using shared storage for the converted checkpoint.
    CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/root}"
    CKPT_ARGS=(
        --hf-checkpoint "/root/models/${MODEL_NAME}"
        --ref-load "${CKPT_SAVE_DIR}/${MODEL_NAME}_torch_dist"
    )
    if [ "$NO_SAVE_OPTIM" -eq 1 ]; then
        CKPT_ARGS+=(--no-save-optim)
    fi

    # --- Rollout ---
    ROLLOUT_ARGS=(
        --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type deepscaler
        --num-rollout 13
        --rollout-batch-size 4
        --n-samples-per-prompt 4
        --rollout-max-response-len 100
        --rollout-temperature 0.8
        --global-batch-size 16
        --balance-data
    )

    # --- Eval ---
    EVAL_ARGS=(
        --eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl
        --n-samples-per-eval-prompt 16
        --eval-max-response-len 16384
        --eval-top-p 0.7
    )

    # --- Training parallelism ---
    # TP=1 (active params only 12B), PP=4 (handle 106B total), EP=8 (128 experts)
    # → DP = 32 / (TP=1 × PP=4) = 8, EP=8 ≤ DP=8 ✓
    PERF_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 4
        --context-parallel-size 1
        --expert-model-parallel-size 8
        --expert-tensor-parallel-size 1
        --decoder-last-pipeline-num-layers 10
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 2048
    )

    # --- GRPO ---
    GRPO_ARGS=(
        --advantage-estimator gspo
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --entropy-coef 0.00
        --eps-clip 4e-4
    )

    # --- Optimizer ---
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

    # --- WANDB ---
    WANDB_ARGS=(
        #--use-wandb
    )

    # --- SGLang: 4 engines × 8 GPUs each ---
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
        --sglang-mem-fraction-static 0.8
        --sglang-ep-size 8
        --sglang-cuda-graph-bs 1 2 4 8 16
        --sglang-enable-dp-attention
        --sglang-enable-dp-lm-head
    )
    if [ "$mode" = "p2p" ]; then
        SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
    fi
    if [ "$SKIP_VALIDATION" -eq 1 ]; then
        SGLANG_ARGS+=(--sglang-load-format dummy)
    else
        SGLANG_ARGS+=(--sglang-model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}')
    fi

    # --- Misc ---
    if [ "$mode" = "p2p" ]; then
        BUFFER_SIZE=$(python3 -c "print(int(${BUCKET_SIZE_GB} * 1024 * 1024 * 1024))")
    else
        BUFFER_SIZE=$((4 * 1024 * 1024 * 1024))
    fi

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --actor-num-nodes ${NUM_TRAIN_NODES}
        --actor-num-gpus-per-node ${GPUS_PER_NODE}
        --update-weight-buffer-size ${BUFFER_SIZE}
    )
    if [ "$SKIP_VALIDATION" -eq 0 ]; then
        MISC_ARGS+=(--check-weight-update-equal)
    fi
    if [ "$mode" = "p2p" ]; then
        MISC_ARGS+=(--update-weight-transfer-mode p2p)
    else
        MISC_ARGS+=(--update-weight-transfer-mode broadcast)
    fi

    # --- Worker nodes sleep to let head node start first ---
    if [ "$NODE_RANK" -gt 0 ]; then
        sleep 20
    fi

    # --- MC transfer timeout ---
    MC_TRANSFER_TIMEOUT=300

    NCCL_NVLS_VAL="0"
    if [ "$ENABLE_NCCL_NVLS" -eq 1 ]; then
        NCCL_NVLS_VAL="1"
    fi

    # --- Launch Ray ---
    # Each node uses its own /tmp/ray (container-local). Ray workers connect
    # to the head via GCS (network), not filesystem.
    if [ "$NODE_RANK" -eq 0 ]; then
        RAY_memory_monitor_refresh_ms=0 \
        ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus ${GPUS_PER_NODE} \
            --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    else
        RAY_memory_monitor_refresh_ms=0 \
        ray start --address="${HEAD_NODE_IP}:6379" --num-gpus ${GPUS_PER_NODE} \
            --disable-usage-stats
    fi

    # --- Build runtime env JSON ---
    RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"MC_TRANSFER_TIMEOUT\": \"${MC_TRANSFER_TIMEOUT}\",
    \"RAY_DEBUG\": \"1\",
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_VAL}\",
    \"MODEL_ARGS_ROTARY_BASE\": \"1000000\",
    \"MILES_LOG_DIR\": \"${MILES_LOG_DIR:-}\"
  }
}"

    # --- Wait for all nodes to join Ray cluster (head node only) ---
    EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Waiting for ${EXPECTED_GPUS} GPUs in Ray cluster..."
        while true; do
            AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null || echo 0)
            echo "  ... detected ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs"
            if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
                break
            fi
            sleep 5
        done
        echo "All ${EXPECTED_GPUS} GPUs available. Submitting job."
    fi

    # --- Signal file for worker synchronization (container env) ---
    SIGNAL_DIR="${MILES_LOG_DIR:-/data/ray/signals}"
    mkdir -p "${SIGNAL_DIR}"
    DONE_FILE="${SIGNAL_DIR}/job_done_${mode}"
    rm -f "${DONE_FILE}"

    # --- Submit Ray job (head node only) ---
    if [ "$NODE_RANK" -eq 0 ]; then
        ray job submit --address="http://127.0.0.1:8265" \
            --runtime-env-json="${RUNTIME_ENV_JSON}" \
            -- python3 train.py \
            ${MODEL_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${EVAL_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} \
            ${GRPO_ARGS[@]} \
            ${WANDB_ARGS[@]} \
            ${PERF_ARGS[@]} \
            ${SGLANG_ARGS[@]} \
            ${MISC_ARGS[@]}
        JOB_EXIT=$?
        echo "${JOB_EXIT}" > "${DONE_FILE}"
    else
        # Worker nodes: block until head node signals completion.
        echo "Worker node ${NODE_RANK}: Ray joined, waiting for head to finish..."
        while [ ! -f "${DONE_FILE}" ]; do
            sleep 10
        done
        echo "Worker node ${NODE_RANK}: head finished (exit=$(cat "${DONE_FILE}")), exiting."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run_mode "$MODE"

echo "Done."
