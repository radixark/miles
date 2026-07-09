#!/bin/bash
#SBATCH --job-name=m3-sft-vlm
#SBATCH --nodes=8
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/m3-sft-vlm-%j.out
#SBATCH --error=./logs/m3-sft-vlm-%j.error
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --exclude=ip-10-1-142-67

# =========================================================================
# MiniMax-M3 (428B) MULTIMODAL full-FT SFT — real LM + real vision tower, on
# image data (geo3k). Mirrors kimi-k2.6/launch_sft_vlm.slurm but for the M3 VL
# stack (--minimax-m3-vl: MiniMaxM3VLModel composite = HF-native vision tower +
# projector + Megatron LM; mm_data expands the <image> placeholder to per-patch
# media tokens; the generic sft_rollout runs the processor -> pixel_values).
#
# Combines the VALIDATED 8-node real-weight infra (EP32, CPU-offload optimizer,
# common-prefix NCCL iface, dynamic-module pre-warm) with the VL bits from
# launch_m3_smoke.slurm MODE=vl (transformers-5.3 shadow, M3_VISION_CKPT).
#
# This is the e2e VLM-training test: vision->project->merge->MSA-LM->loss->
# backward->Adam step, real weights, multinode. EP32 needs world%32==0 -> 8 nodes.
#
# Knobs:  NUM_ROLLOUT=20  LR=1e-5  SAVE_INTERVAL=0(off)  MAX_TOKENS_PER_GPU=8192
#         TP=8 PP=1 EP=32 ETP=1
# Submit: cd <miles>; mkdir -p logs; sbatch examples/minimax_m3_vl/launch_sft_vlm_m3.slurm
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-2/miles}
HF_CKPT=${HF_CKPT:-/fsx/peng/models/MiniMax-M3}                 # full 60-layer config + tokenizer + processor
LOAD_HOST=${LOAD_HOST:-/fsx/peng/models/MiniMax-M3_torch_dist}  # converted LM weights (torch_dist)
VISION_CKPT_HOST=${VISION_CKPT_HOST:-/fsx/peng/models/MiniMax-M3}  # vision tower + projector safetensors
TRANSFORMERS_HOST=${TRANSFORMERS_HOST:-/fsx/peng/workspace/transformers}  # has in-library minimax_m3_vl
DATASETS_HOST=${DATASETS_HOST:-${MILES_HOST}/datasets}
SAVE_HOST=${SAVE_HOST:-/fsx/peng/models/MiniMax-M3_sft_vlm_ckpt}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

MILES_ROOT=/workspace/miles; HF_ROOT=/root/m3_hf; LOAD_ROOT=/root/m3_load
DATASETS_ROOT=/root/datasets; SAVE_ROOT=/root/m3_save
TRANSFORMERS_ROOT=/root/transformers_m3; VISION_CKPT_ROOT=/root/m3_vision
TP=${TP:-8}; PP=${PP:-1}; EP=${EP:-32}; ETP=${ETP:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-20}; LR=${LR:-1e-5}; MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
TRAIN_VISION=${TRAIN_VISION:-0}   # 1 = also train the vision tower (default frozen)
VL_PROBE=${VL_PROBE:-0}           # 1 = log vision/projector weight norms per TP rank
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

# Ray session dir on the big local NVMe (not small /tmp, which fills on some nodes;
# not Lustre, which has no Unix sockets). Mirrors the Kimi VLM launcher.
RAY_TMP_HOST=/opt/dlami/nvme/ray-tmp-${USER:-peng}-${SLURM_JOB_ID:-manual}; RAY_TMP_ROOT=/tmp/ray-session
srun --overlap --ntasks-per-node=1 bash -c "mkdir -p ${RAY_TMP_HOST}"
[[ "${SAVE_INTERVAL}" != "0" ]] && mkdir -p "${SAVE_HOST}"
SAVE_MOUNT=""; [[ "${SAVE_INTERVAL}" != "0" ]] && SAVE_MOUNT="${SAVE_HOST}:${SAVE_ROOT},"
# VL needs transformers-5.3 (in-library minimax_m3_vl) shadowed via PYTHONPATH, and
# the vision weights loaded by build_minimax_m3_vl from M3_VISION_CKPT.
TF_PP="${TRANSFORMERS_ROOT}/src:"
MOUNTS="${MILES_HOST}:${MILES_ROOT},${HF_CKPT}:${HF_ROOT},${LOAD_HOST}:${LOAD_ROOT},\
${DATASETS_HOST}:${DATASETS_ROOT},${TRANSFORMERS_HOST}:${TRANSFORMERS_ROOT},${VISION_CKPT_HOST}:${VISION_CKPT_ROOT},\
${SAVE_MOUNT}${RAY_TMP_HOST}:${RAY_TMP_ROOT},/var/log/aws/clusters:/var/log/aws/clusters"

# ---------- cluster discovery (per-node iface -> common prefix; NICs differ per node) ----------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${nodes[0]}
common_prefix() { local prefix=$1; shift; local value
  for value in "$@"; do while [[ -n "${prefix}" && "${value#"$prefix"}" == "${value}" ]]; do prefix=${prefix%?}; done; done
  printf '%s' "${prefix}"; }
ALL_IFACES=()
for node in "${nodes[@]}"; do
  ni=$(srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc "ip route show default 0.0.0.0/0 | awk 'NR==1{print \$5}'")
  [[ -z "${ni}" ]] && { echo "Failed to detect iface for ${node}" >&2; exit 1; }
  ALL_IFACES+=("${ni}")
done
IFACE=$(common_prefix "${ALL_IFACES[0]}" "${ALL_IFACES[@]:1}")
[[ -z "${IFACE}" ]] && IFACE="^lo,docker,veth,tailscale"
HEAD_IFACE=${ALL_IFACES[0]}
echo "== node ifaces: ${ALL_IFACES[*]} -> NCCL_SOCKET_IFNAME=${IFACE} =="
MASTER_ADDR=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "ip -o -4 addr show dev ${HEAD_IFACE} | awk '{print \$4}' | cut -d/ -f1 | head -n1")
MASTER_PORT=6379
NVLINK_COUNT=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9]*' | wc -l" || echo 0)
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1 || HAS_NVLINK=0
ALL_HOSTS_CSV=$(IFS=,; echo "${nodes[*]}")
export NCCL_SOCKET_IFNAME="${IFACE}" NCCL_DEBUG=WARN CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_PROTO="simple" NCCL_SOCKET_FAMILY="AF_INET"
export RDMAV_FORK_SAFE=1 FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${ALL_HOSTS_CSV}"
echo "== M3 VLM SFT | head=${HEAD_NODE} (${MASTER_ADDR}) | nodes=${#nodes[@]} | TP${TP} PP${PP} EP${EP} =="

cleanup(){ local ec=$?; trap - EXIT INT TERM; set +e
  for n in "${nodes[@]}"; do srun --overlap --nodes=1 --ntasks=1 -w "$n" --container-image "${IMAGE}" \
    --container-mounts "${MOUNTS}" --container-remap-root bash -lc "ray stop --force||true"||true; done; exit "$ec"; }
trap cleanup EXIT INT TERM
PIDS=()

# Pre-warm the trust_remote_code dynamic-module cache once per node (avoids the
# 8-actors/node race: AttributeError ... has no attribute 'MiniMaxM3VLConfig').
PREWARM="python3 -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('${HF_ROOT}', trust_remote_code=True)\" >/dev/null 2>&1 || true"

# ---------- ray head ----------
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
  bash -lc "set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
    ${PREWARM}
    ray stop --force||true
    ray start --head --node-ip-address=${MASTER_ADDR} --port=${MASTER_PORT} --num-gpus ${GPUS_PER_NODE} \
      --temp-dir=${RAY_TMP_ROOT} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block" &
PIDS+=($!); sleep 20

# ---------- ray workers ----------
for ((i=1;i<${#nodes[@]};i++)); do
  srun --nodes=1 --ntasks=1 -w "${nodes[$i]}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
    bash -lc "set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
      pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
      ${PREWARM}
      ray stop --force||true
      ray start --address ${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${GPUS_PER_NODE} --temp-dir=${RAY_TMP_ROOT} --disable-usage-stats --block" &
  PIDS+=($!); sleep 5
done
sleep 30
for a in $(seq 1 36); do timeout 2 bash -c "</dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null && break; sleep 5; done

# ---------- submit ----------
set +e
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" \
  --container-remap-root --container-env=NCCL_SOCKET_IFNAME,NCCL_DEBUG,no_proxy \
  bash -lc "
    set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
    source scripts/models/minimax-m3-428B.sh
    [[ -f ${DATASETS_ROOT}/geo3k_imgurl/train_formatted.parquet ]] || \
      hf download --repo-type dataset chenhegu/geo3k_imgurl --local-dir ${DATASETS_ROOT}/geo3k_imgurl || true
    OPT=( --optimizer adam --lr ${LR} --lr-decay-style constant --weight-decay 0.0 \
          --optimizer-cpu-offload --optimizer-offload-fraction 1.0 --overlap-cpu-optimizer-d2h-h2d \
          --use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 )
    SAVE_FLAGS=()
    [[ '${SAVE_INTERVAL}' != '0' ]] && SAVE_FLAGS=( --save ${SAVE_ROOT} --save-interval ${SAVE_INTERVAL} --no-save-optim )
    VL_FLAGS=( --minimax-m3-vl )
    [[ '${TRAIN_VISION}' == '1' ]] && VL_FLAGS+=( --minimax-m3-train-vision )
    ray job submit --address=http://${MASTER_ADDR}:8265 \
      --runtime-env-json='{\"env_vars\":{\"PYTHONPATH\":\"${TF_PP}${MILES_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"RAY_DEDUP_LOGS\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"M3_VISION_CKPT\":\"${VISION_CKPT_ROOT}\",\"M3_VL_PROBE\":\"${VL_PROBE}\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"NCCL_SOCKET_IFNAME\":\"${IFACE}\",\"NCCL_PROTO\":\"simple\",\"NCCL_SOCKET_FAMILY\":\"AF_INET\",\"RDMAV_FORK_SAFE\":\"1\",\"FI_EFA_USE_DEVICE_RDMA\":\"1\",\"FI_EFA_SET_CUDA_SYNC_MEMOPS\":\"0\",\"TORCH_NCCL_ASYNC_ERROR_HANDLING\":\"1\",\"NCCL_DEBUG\":\"WARN\",\"no_proxy\":\"${no_proxy}\"}}' \
      -- python3 train.py --actor-num-nodes ${#nodes[@]} --actor-num-gpus-per-node ${GPUS_PER_NODE} \
      \"\${MODEL_ARGS[@]}\" \"\${VL_FLAGS[@]}\" \
      --hf-checkpoint ${HF_ROOT} --load ${LOAD_ROOT} \
      --prompt-data ${DATASETS_ROOT}/geo3k_imgurl/train_formatted.parquet --input-key messages \
      --multimodal-keys '{\"image\": \"images\"}' \
      --rollout-function-path miles.rollout.sft_rollout.generate_rollout --rollout-shuffle \
      --num-rollout ${NUM_ROLLOUT} --rollout-batch-size 16 --global-batch-size 16 \
      --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only \
      --tensor-model-parallel-size ${TP} --sequence-parallel --pipeline-model-parallel-size ${PP} \
      --context-parallel-size 1 --expert-model-parallel-size ${EP} --expert-tensor-parallel-size ${ETP} \
      --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
      --use-dynamic-batch-size --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
      --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 \
      --attention-softmax-in-fp32 --attention-backend flash \
      \"\${OPT[@]}\" \"\${SAVE_FLAGS[@]}\"
  " 2>&1 | tee ./logs/m3-sft-vlm-${SLURM_JOB_ID}.log
ec=${PIPESTATUS[0]}; set -e
[[ "${ec}" -ne 0 ]] && { echo "M3 VLM SFT FAILED (exit ${ec})"; exit "${ec}"; }
echo "M3 VLM SFT completed."
