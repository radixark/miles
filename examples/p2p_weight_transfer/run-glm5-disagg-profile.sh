#!/bin/bash

# Unified multi-node disaggregated profiling script for all GLM-5 variants.
# Supports broadcast and p2p weight transfer modes.
#
# Configurations (from run_glm5_744b_a40b.py):
#   GLM-5_4layer   2 nodes  (1 train + 1 rollout)   TP=4 PP=1 EP=8
#   GLM-5_20layer  12 nodes (6 train + 6 rollout)   TP=4 PP=3 EP=16
#   GLM-5          32 nodes (16 train + 16 rollout)  TP=4 PP=4 CP=2 EP=32
#
# Usage:
#   bash run-glm5-disagg-profile.sh <MODEL_NAME> <MODE> <NODE_RANK> <HEAD_NODE_IP>
#
#   MODEL_NAME    : GLM-5_4layer | GLM-5_20layer | GLM-5
#   MODE          : broadcast | p2p
#   NODE_RANK     : 0 (head node) | 1..N (worker nodes)
#   HEAD_NODE_IP  : IP address of the head node
#
# Examples:
#   bash run-glm5-disagg-profile.sh GLM-5_4layer  broadcast 0 10.0.0.1
#   bash run-glm5-disagg-profile.sh GLM-5_20layer p2p       0 10.0.0.1
#   bash run-glm5-disagg-profile.sh GLM-5         p2p       0 10.0.0.1

set -ex

export PYTHONBUFFERED=16

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ $# -lt 4 ]; then
    echo "Usage: $0 <MODEL_NAME> <MODE> <NODE_RANK> <HEAD_NODE_IP>"
    echo "  MODEL_NAME    : GLM-5_4layer | GLM-5_20layer | GLM-5"
    echo "  MODE          : broadcast | p2p"
    echo "  NODE_RANK     : 0 (head) | 1..N (workers)"
    echo "  HEAD_NODE_IP  : IP of the head node"
    exit 1
fi

MODEL_NAME="$1"
MODE="$2"
NODE_RANK="$3"
HEAD_NODE_IP="$4"

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
# Model-specific config
# ---------------------------------------------------------------------------
GPUS_PER_NODE=8
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
BUCKET_SIZE_GB="${BUCKET_SIZE_GB:-1.0}"
ENABLE_NCCL_NVLS=1

case "${MODEL_NAME}" in
    GLM-5_4layer)
        MODEL_TYPE="glm5-744B-A40B_4layer"
        NNODES=2
        NUM_TRAIN_GPUS=8      # 1 node
        NUM_ROLLOUT_GPUS=8    # 1 node
        # Training parallelism: TP=4, PP=1, CP=1, EP=8
        TP=4; PP=1; CP=1; EP=8
        DECODER_LAST_PIPELINE=""
        MAX_TOKENS_PER_GPU=2048
        SGLANG_WORLD_SIZE=8
        SGLANG_DECODE_MAX_BS=8
        SGLANG_MEM_FRAC=0.70
        ENABLE_OPTIMIZER_OFFLOAD=0
        ;;
    GLM-5_20layer)
        MODEL_TYPE="glm5-744B-A40B_20layer"
        NNODES=12
        NUM_TRAIN_GPUS=48     # 6 nodes
        NUM_ROLLOUT_GPUS=48   # 6 nodes
        # Training parallelism: TP=4, PP=3, CP=1, EP=16 (from run_glm5_744b_a40b.py 6-node)
        TP=4; PP=3; CP=1; EP=16
        DECODER_LAST_PIPELINE="--decoder-last-pipeline-num-layers 6"
        MAX_TOKENS_PER_GPU=1024
        SGLANG_WORLD_SIZE=16
        SGLANG_DECODE_MAX_BS=8
        SGLANG_MEM_FRAC=0.70
        ENABLE_OPTIMIZER_OFFLOAD=1
        ;;
    GLM-5)
        MODEL_TYPE="glm5-744B-A40B"
        NNODES=32
        NUM_TRAIN_GPUS=128    # 16 nodes
        NUM_ROLLOUT_GPUS=128  # 16 nodes
        # Training parallelism: TP=4, PP=8, CP=2, EP=16 (PP doubled to reduce per-GPU memory)
        # EP reduced from 32 to 16 since EP*PP must divide world_size (128)
        TP=4; PP=8; CP=2; EP=16
        DECODER_LAST_PIPELINE="--decoder-last-pipeline-num-layers 8"
        MAX_TOKENS_PER_GPU=256
        SGLANG_WORLD_SIZE=64
        SGLANG_DECODE_MAX_BS=8
        SGLANG_MEM_FRAC=0.90
        ENABLE_OPTIMIZER_OFFLOAD=1
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL_NAME}'. Use GLM-5_4layer, GLM-5_20layer, or GLM-5."
        exit 1
        ;;
esac

NUM_TRAIN_NODES=$((NUM_TRAIN_GPUS / GPUS_PER_NODE))

MILES_ROOT="/root/miles"
source "${MILES_ROOT}/scripts/models/${MODEL_TYPE}.sh"

echo ""
echo "============================================================"
echo "  Model      : ${MODEL_NAME} (${MODEL_TYPE})"
echo "  Mode       : ${MODE}"
echo "  Nodes      : ${NNODES} total (${NUM_TRAIN_NODES} train + $((NNODES - NUM_TRAIN_NODES)) rollout)"
echo "  Parallelism: TP=${TP} PP=${PP} CP=${CP} EP=${EP}"
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
        --rollout-temperature 1
        --global-batch-size 16
    )

    # --- Training parallelism ---
    PERF_ARGS=(
        --tensor-model-parallel-size ${TP}
        --sequence-parallel
        --pipeline-model-parallel-size ${PP}
        --context-parallel-size ${CP}
        --expert-model-parallel-size ${EP}
        --expert-tensor-parallel-size 1
        ${DECODER_LAST_PIPELINE}
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
        --data-pad-size-multiplier 4096
        --log-probs-chunk-size 1024
    )

    # --- GRPO ---
    GRPO_ARGS=(
        --advantage-estimator grpo
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --kl-coef 0.00
        --entropy-coef 0.00
        --eps-clip 0.2
        --eps-clip-high 0.28
    )

    # --- Optimizer ---
    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr 1e-6
        --lr-decay-style constant
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
    )
    if [ "$ENABLE_OPTIMIZER_OFFLOAD" -eq 1 ]; then
        OPTIMIZER_ARGS+=(
            --optimizer-cpu-offload
            --overlap-cpu-optimizer-d2h-h2d
            --use-precision-aware-optimizer
        )
    fi

    # --- WANDB ---
    WANDB_ARGS=(
        #--use-wandb
    )

    # --- SGLang ---
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine ${SGLANG_WORLD_SIZE}
        --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
        --sglang-mem-fraction-static ${SGLANG_MEM_FRAC}
        --sglang-enable-dp-attention
        --sglang-ep-size ${SGLANG_WORLD_SIZE}
        --sglang-dp-size ${SGLANG_WORLD_SIZE}
        --sglang-moe-dense-tp-size 1
        --sglang-enable-dp-lm-head
        # GLM5 NSA attention and chunked prefill
        --sglang-page-size 64
        --sglang-nsa-decode-backend flashmla_sparse
        --sglang-nsa-prefill-backend flashmla_sparse
        --sglang-attention-backend nsa
        --sglang-cuda-graph-max-bs ${SGLANG_DECODE_MAX_BS}
        --sglang-max-running-requests 512
        --sglang-chunked-prefill-size $((2048 * ${SGLANG_WORLD_SIZE}))
        --sglang-watchdog-timeout 3600
        --sglang-disable-cuda-graph
    )
    if [ "$mode" = "p2p" ]; then
        SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
        # Optional: pin each GPU rank to its own IB HCA for optimal RDMA locality.
        # Not required for correctness — validated that auto-discovery (all NICs) works
        # without MTT overflow on GLM-5 744B (32 nodes). Kept for potential perf benefit.
        SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-ib-device '{"0":"ibp0","1":"ibp1","2":"ibp2","3":"ibp3","4":"ibp4","5":"ibp5","6":"ibp6","7":"ibp7"}')
    fi
    if [ "$SKIP_VALIDATION" -eq 1 ]; then
        SGLANG_ARGS+=(--sglang-load-format dummy)
    else
        SGLANG_ARGS+=(--sglang-model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}')
    fi

    # --- Misc ---
    BUFFER_SIZE="${BUFFER_SIZE:-$((2 * 1024 * 1024 * 1024))}"

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --allgather-cp
        --moe-token-dispatcher-type alltoall
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
    MC_TRANSFER_TIMEOUT=600

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
    \"INDEXER_ROPE_NEOX_STYLE\": \"0\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
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
    # In container environments, worker nodes must stay alive while the
    # head node runs the Ray job. We use a signal file on shared storage.
    SIGNAL_DIR="${MILES_LOG_DIR:-/data/ray/signals}"
    mkdir -p "${SIGNAL_DIR}"
    DONE_FILE="${SIGNAL_DIR}/job_done_${mode}"

    # Clean up any stale signal file
    rm -f "${DONE_FILE}"

    # --- Submit Ray job (head node only) ---
    if [ "$NODE_RANK" -eq 0 ]; then
        # Use || true to prevent set -e from exiting before writing DONE_FILE
        ray job submit --address="http://127.0.0.1:8265" \
            --runtime-env-json="${RUNTIME_ENV_JSON}" \
            -- python3 train.py \
            ${MODEL_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} \
            ${GRPO_ARGS[@]} \
            ${WANDB_ARGS[@]} \
            ${PERF_ARGS[@]} \
            ${SGLANG_ARGS[@]} \
            ${MISC_ARGS[@]} \
            || true
        JOB_EXIT=$?
        # Signal workers that the job is done
        echo "${JOB_EXIT}" > "${DONE_FILE}"
    else
        # Worker nodes: block until head node signals completion.
        # In container environments (pyxis/enroot), exiting kills the container
        # and the Ray worker with it, so we must stay alive.
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
