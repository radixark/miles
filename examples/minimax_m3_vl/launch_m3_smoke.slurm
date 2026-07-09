#!/bin/bash
#SBATCH --job-name=m3-smoke
#SBATCH --nodes=1
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/m3-smoke-%j.out
#SBATCH --error=./logs/m3-smoke-%j.err
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --wait-all-nodes=1
# Nodes with broken pyxis (container-mounts silently no-op there).
#SBATCH --exclude=ip-10-1-8-7,ip-10-1-7-180,ip-10-1-38-198,ip-10-1-142-67,ip-10-1-126-226,ip-10-1-141-192,ip-10-1-94-38

# =========================================================================
# MiniMax-M3 SINGLE-NODE plumbing smoke test (item 1).
#
# Validates the NEW M3 code on 4 random-init layers (no real LM weights), via
# --debug-train-only SFT (no sglang/rollout engines). Two modes:
#
#   MODE=text  (default) — cheapest, NO big checkpoint needed. Exercises the
#       risky novel path: get_minimax_m3_spec build -> MSA lightning indexer +
#       block-sparse attention forward -> MoE -> loss/backward. Tiny text data.
#
#   MODE=vl    — adds --minimax-m3-vl: composite model + HF-native vision tower
#       + mm_data media-token expansion + embed/merge/decoder_input. Needs the
#       M3 HF checkpoint present (HF_CKPT) so build_minimax_m3_vl can load the
#       vision tower + processor, and the geo3k image dataset.
#
# Resubmit knobs (env vars):
#   IMAGE=/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh
#   MILES_HOST=/fsx/peng/workspace/miles-oss-workspace-2/miles
#   MODE=text|vl   HF_CKPT=/fsx/peng/models/MiniMax-M3
#   TP=8 PP=1 EP=8   MAX_TOKENS_PER_GPU=4096
#
# Submit:  cd <miles>; mkdir -p logs; sbatch examples/minimax_m3_vl/launch_m3_smoke.slurm
#     VL:  MODE=vl HF_CKPT=/fsx/peng/models/MiniMax-M3 sbatch examples/minimax_m3_vl/launch_m3_smoke.slurm
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

MODE=${MODE:-text}

# ---------- Image ----------
IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
if [[ ! -f "${IMAGE}" ]]; then echo "ERROR: IMAGE not found: ${IMAGE}" >&2; exit 1; fi

# ---------- Paths ----------
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-2/miles}
DATASETS_HOST=${DATASETS_HOST:-${MILES_HOST}/datasets}
OUTPUTS_HOST=${OUTPUTS_HOST:-${MILES_HOST}/outputs}
# M3 HF checkpoint = 4-layer config dir for BOTH modes: --hf-checkpoint feeds
# hf_validate_args (num_layers must match the 4-layer model args) + tokenizer +
# vision_config. In vl mode the vision *weights* come from the full ckpt
# (VISION_CKPT_HOST) loaded separately by build_minimax_m3_vl (M3_VISION_CKPT).
# MODE=real: full 60-layer model loaded from the converted torch_dist checkpoint
# (validates conversion). --hf-checkpoint = full ckpt (60-layer config matches).
if [[ "${MODE}" == "real" ]]; then
  HF_CKPT=${HF_CKPT:-/fsx/peng/models/MiniMax-M3}
  LOAD_HOST=${LOAD_HOST:-/fsx/peng/models/MiniMax-M3_torch_dist}
else
  HF_CKPT=${HF_CKPT:-/fsx/peng/models/MiniMax-M3-4layer}
  LOAD_HOST=""
fi
LOAD_ROOT=/root/m3_load
VISION_CKPT_HOST=${VISION_CKPT_HOST:-/fsx/peng/models/MiniMax-M3}
mkdir -p "${DATASETS_HOST}" "${OUTPUTS_HOST}"

MILES_ROOT=/workspace/miles
DATASETS_ROOT=/root/datasets
OUTPUTS_ROOT=/workspace/outputs
HF_ROOT=/root/m3_hf

# VL mode needs transformers with the in-library `minimax_m3_vl` model (added
# 2026-06-12). The container's transformers predates it, so mount a checkout that
# has it (pure Python -> shadow via PYTHONPATH). Gated to VL so the working text
# path keeps the container's transformers. Override TRANSFORMERS_HOST if needed.
TRANSFORMERS_HOST=${TRANSFORMERS_HOST:-/fsx/peng/workspace/transformers}
TRANSFORMERS_ROOT=/root/transformers_m3
VISION_CKPT_ROOT=/root/m3_vision
if [[ "${MODE}" == "vl" ]]; then
  TF_MOUNT="${TRANSFORMERS_HOST}:${TRANSFORMERS_ROOT},${VISION_CKPT_HOST}:${VISION_CKPT_ROOT},"
  TF_PP="${TRANSFORMERS_ROOT}/src:"   # prepended to PYTHONPATH -> shadows installed transformers
  M3_VISION_ENV="${VISION_CKPT_ROOT}"  # build_minimax_m3_vl reads M3_VISION_CKPT for vision weights
else
  TF_MOUNT=""
  TF_PP=""
  M3_VISION_ENV=""
fi

# Ray session dir must be local (Lustre has no Unix sockets); /tmp is local on EC2.
RAY_TMP_HOST=${RAY_TMP_HOST:-/tmp/ray-tmp-${USER:-peng}-${SLURM_JOB_ID:-manual}}
RAY_TMP_ROOT=/tmp/ray-session
srun --overlap --ntasks-per-node=1 bash -c "mkdir -p ${RAY_TMP_HOST}"

LOAD_MOUNT=""
[[ -n "${LOAD_HOST}" ]] && LOAD_MOUNT="${LOAD_HOST}:${LOAD_ROOT},"
MILES_MOUNTS="${MILES_HOST}:${MILES_ROOT},\
${DATASETS_HOST}:${DATASETS_ROOT},\
${OUTPUTS_HOST}:${OUTPUTS_ROOT},\
${HF_CKPT}:${HF_ROOT},\
${LOAD_MOUNT}${TF_MOUNT}\
${RAY_TMP_HOST}:${RAY_TMP_ROOT},\
/var/log/aws/clusters:/var/log/aws/clusters"

# ---------- Run knobs ----------
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
if [[ "${MODE}" == "real" ]]; then
  # Full 60-layer model. TP8/PP1 (not PP8: 60 not divisible by 8, and torch_dist
  # reshards the PP8-saved checkpoint to TP8 on load). 428B/8 ~= 107GB/GPU fits.
  TP=${TP:-8}; PP=${PP:-1}; EP=${EP:-1}
  MODEL_SCRIPT=scripts/models/minimax-m3-428B.sh
else
  TP=${TP:-8}; PP=${PP:-1}; EP=${EP:-8}
  MODEL_SCRIPT=scripts/models/minimax-m3-428B_4layer.sh
fi
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}

# ---------- Cluster discovery (single node) ----------
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
detect_iface() { srun --overlap --nodes=1 --ntasks=1 -w "$1" bash -lc "ip route show default | awk 'NR==1{print \$5}'"; }
node_ip() { srun --overlap --nodes=1 --ntasks=1 -w "$1" bash -lc "ip -o -4 addr show dev $2 | awk '{print \$4}' | cut -d/ -f1 | head -n1"; }
IFACE=$(detect_iface "${HEAD_NODE}"); [[ -z "${IFACE}" ]] && IFACE=eth0
MASTER_ADDR=$(node_ip "${HEAD_NODE}" "${IFACE}")
MASTER_PORT=6379
NVLINK_COUNT=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9]*' | wc -l" || echo 0)
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "== M3 smoke | MODE=${MODE} | head=${HEAD_NODE} (${MASTER_ADDR}) | TP${TP} PP${PP} EP${EP} | nvlink=${HAS_NVLINK} =="

# ---------- Env ----------
export NCCL_SOCKET_IFNAME="${IFACE}"
export NCCL_DEBUG="WARN"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR},${HEAD_NODE}"
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=${WANDB_API_KEY:-}

cleanup() {
  local ec=$?; trap - EXIT INT TERM; set +e
  srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
    bash -lc "ray stop --force || true" || true
  exit "${ec}"
}
trap cleanup EXIT INT TERM

########################################
# 1. Ray head (single node)
########################################
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
  --container-env=WANDB_API_KEY \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3
    ray stop --force || true
    ray start --head --node-ip-address=${MASTER_ADDR} --port=${MASTER_PORT} \
      --num-gpus ${GPUS_PER_NODE} --disable-usage-stats \
      --temp-dir=${RAY_TMP_ROOT} --dashboard-host=0.0.0.0 --dashboard-port=8265 --block
  " &
sleep 25
for a in $(seq 1 24); do
  timeout 2 bash -c "</dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null && break
  [[ "$a" == 24 ]] && { echo "ERROR: Ray GCS never came up" >&2; exit 1; }
  sleep 5
done

########################################
# 2. Submit the smoke training job
########################################
set +e
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
  --container-env=WANDB_API_KEY,NCCL_SOCKET_IFNAME,NCCL_DEBUG,no_proxy \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3

    source ${MODEL_SCRIPT}

    # ---- mode-specific data + VL flag ----
    MODE=${MODE}
    if [[ \"\${MODE}\" == \"vl\" ]]; then
      # geo3k images (same dataset as examples/geo3k_vlm)
      if [[ ! -f ${DATASETS_ROOT}/geo3k_imgurl/train_formatted.parquet ]]; then
        hf download --repo-type dataset chenhegu/geo3k_imgurl --local-dir ${DATASETS_ROOT}/geo3k_imgurl || true
      fi
      DATA_ARGS=(
        --prompt-data ${DATASETS_ROOT}/geo3k_imgurl/train_formatted.parquet
        --input-key messages
        --multimodal-keys '{\"image\": \"images\"}'
      )
      MODE_ARGS=( --minimax-m3-vl )
    else
      # tiny SYNTHETIC text SFT data in messages format (pre-generated on the host
      # at \${MILES_HOST}/datasets/m3_smoke_text.parquet -> \${DATASETS_ROOT}). The
      # SFT mask generator needs a list of {role,content} dicts; plain 'prompt'
      # strings (dapo-math) crash it.
      # NO --apply-chat-template here: the sft_loss mask generator wants a list of
      # role/content dicts and templates each message itself; pre-rendering to a
      # string makes it iterate characters and crash.
      DATA_ARGS=(
        --prompt-data ${DATASETS_ROOT}/m3_smoke_text.parquet
        --input-key messages
      )
      MODE_ARGS=()
    fi

    if [[ \"\${MODE}\" == \"real\" ]]; then
      # Load the converted torch_dist weights; disable optimizer so the full
      # 428B fits on one node for a load + forward/backward check. A CORRECT
      # conversion -> LOW loss (~1-3); random/garbage -> high (~7+).
      CKPT_ARGS=(
        --hf-checkpoint ${HF_ROOT}
        --load ${LOAD_ROOT}
        --debug-disable-optimizer
      )
    else
      CKPT_ARGS=(
        --hf-checkpoint ${HF_ROOT}    # tokenizer/config (+ vision tower in vl mode)
        --debug-skip-load             # RANDOM init, no checkpoint (plumbing smoke)
      )
    fi
    SFT_ARGS=(
      --rollout-function-path miles.rollout.sft_rollout.generate_rollout
      --rollout-shuffle
      --num-rollout 2                 # 2 train steps then exit — smoke only
      --rollout-batch-size 8
      --global-batch-size 8
      --loss-type sft_loss
      --calculate-per-token-loss
      --disable-compute-advantages-and-returns
      --debug-train-only
    )
    PERF_ARGS=(
      --tensor-model-parallel-size ${TP}
      --sequence-parallel
      --pipeline-model-parallel-size ${PP}
      --context-parallel-size 1
      --expert-model-parallel-size ${EP}
      --expert-tensor-parallel-size 1
      --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
    )
    OPT_ARGS=( --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.0 )
    MISC_ARGS=(
      --attention-dropout 0.0 --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
    )

    ray job submit --address=http://${MASTER_ADDR}:8265 \
      --runtime-env-json='{\"env_vars\":{\"PYTHONPATH\":\"${TF_PP}${MILES_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"RAY_DEDUP_LOGS\":\"0\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"M3_VISION_CKPT\":\"${M3_VISION_ENV}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}}' \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node ${GPUS_PER_NODE} \
      \"\${MODEL_ARGS[@]}\" \"\${MODE_ARGS[@]}\" \"\${CKPT_ARGS[@]}\" \"\${DATA_ARGS[@]}\" \
      \"\${SFT_ARGS[@]}\" \"\${PERF_ARGS[@]}\" \"\${OPT_ARGS[@]}\" \"\${MISC_ARGS[@]}\"
  " 2>&1 | tee ./logs/m3-smoke-${SLURM_JOB_ID}.log
ec=${PIPESTATUS[0]}
set -e
[[ "${ec}" -ne 0 ]] && { echo "M3 smoke FAILED (exit ${ec})"; exit "${ec}"; }
echo "M3 smoke completed."
