#!/bin/bash
# MiniMax-M3-VL SFT — end-to-end smoke/training launcher.
#
# Exercises the full VL path added in miles_plugins/models/minimax_m3:
#   --spec get_minimax_m3_spec        : MSA + MoE text decoder (the language_model)
#   --minimax-m3-vl                   : wrap it in the composite VL model
#                                       (HF-native vision tower + projector)
#   mm_data token-expansion           : runs automatically in get_rollout_data
#
# Build/load follows the GLM-5 pattern (custom-attention models), NOT kimi's
# bridge-build: --spec builds the MSA model; --megatron-to-hf-mode stays "raw"
# (default) so model_provider uses the --spec path (bridge mode would bypass it).
# Weights load from an OFFLINE-converted Megatron checkpoint like glm5 does (see
# scripts/run_glm5_744b_a40b.py convert_checkpoint + --ref-load *_torch_dist);
# the MiniMaxM3Bridge mappings drive that conversion. The HF vision tower loads
# directly via build_minimax_m3_vl regardless.
#
# Modeled on examples/geo3k_vlm/run_geo3k_vlm_sft.sh (Qwen3-VL) for the harness,
# but the model-build path mirrors glm5. Uses the geo3k image dataset.
#
# NOTE: MiniMax-M3 is a ~428B MoE — this needs multi-node + heavy parallelism.
# For a *plumbing smoke test* without the full weights, see SMOKE notes at the
# bottom (reduced layers + skip bridge load).

TRAIN_BACKEND=${MILES_SCRIPT_TRAIN_BACKEND:-"megatron"}
MODEL_NAME=${MILES_SCRIPT_MODEL_NAME:-"MiniMax-M3"}
HF_REPO=${MILES_SCRIPT_HF_REPO:-"MiniMaxAI/MiniMax-M3"}
DATASET_NAME=${MILES_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${MILES_SCRIPT_NUM_GPUS:-8}
NUM_NODES=${MILES_SCRIPT_NUM_NODES:-4}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")
MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$MILES_SCRIPT_EXTERNAL_RAY" ] || [ "$MILES_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang; sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then ray stop --force; pkill -9 ray; fi
pkill -9 miles; sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then pkill -9 ray; fi
pkill -9 miles; pkill -9 redis

set -ex
export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model + dataset
mkdir -p /root/models /root/datasets
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   hf download ${HF_REPO} --local-dir /root/models/${MODEL_NAME}
fi
if [ ! -d "/root/datasets/${DATASET_LOCAL_NAME}" ]; then
   hf download --repo-type dataset ${DATASET_NAME} --local-dir /root/datasets/${DATASET_LOCAL_NAME}
fi

CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --load /root/models/${MODEL_NAME}
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data /root/datasets/${DATASET_LOCAL_NAME}/train_formatted.parquet
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 32
   --global-batch-size 32

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# tell the data loader which message field carries images
MULTIMODAL_KEYS='{"image": "images"}'

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project miles-minimax-m3-vl-sft
        --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
        --wandb-key ${WANDB_API_KEY}
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

# Megatron backend (M3 has MSA + custom indexer -> megatron only, no fsdp path)
BACKEND_ARGS=(
   --train-backend megatron
   # ---- parallelism for a ~428B MoE; tune to your cluster ----
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1            # MSA reference path assumes CP=1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # NOTE: leave --megatron-to-hf-mode at its default ("raw"). Setting "bridge"
   # makes model_provider build via the bridge provider and bypass --spec (no
   # MSA). M3 builds via --spec like glm5; weights come from --load (offline
   # convert HF->megatron first, mirroring scripts/run_glm5_744b_a40b.py).
)

# M3 model args (hidden/layers/heads/MoE/--spec) + the VL switch
MILES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
source "${MILES_DIR}/scripts/models/minimax-m3-428B.sh"
M3_VL_ARGS=(
   --minimax-m3-vl                      # wrap text decoder in the composite VL model
)

# Start Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${M3_VL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]}

# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST (plumbing only, no real weights / few GPUs):
#   1. Make a reduced model arg file scripts/models/minimax-m3-428B_4layer.sh
#      (copy minimax-m3-428B.sh, set --num-layers 4 and N_MOE_LAYERS=1) and
#      `source` it instead above.
#   2. Drop weight loading so the layer-count mismatch doesn't abort: remove the
#      `--load` line (random-init LM). The HF vision tower still loads real
#      weights via build_minimax_m3_vl, which is what exercises the VL merge path.
#   3. Shrink parallelism (TP=1/2, PP=1, EP=1) and NUM_GPUS to fit one node.
#   This validates: spec build -> composite wrap -> mm_data expansion ->
#   embed/merge/decoder_input forward, end to end, on a single node.
# ─────────────────────────────────────────────────────────────────────────────
