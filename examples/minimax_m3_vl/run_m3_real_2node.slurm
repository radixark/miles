#!/bin/bash
#SBATCH --job-name=m3-real
#SBATCH --nodes=2
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/m3-real-%j.out
#SBATCH --error=./logs/m3-real-%j.err
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --exclude=ip-10-1-142-67

# =========================================================================
# MiniMax-M3 REAL-WEIGHT run on 2 nodes (16 GPUs). Loads the converted
# torch_dist checkpoint into the full 60-layer model and does SFT.
# 428B/16 ~= 53.5GB params/GPU -> fits (1 node OOMs: 107GB/GPU).
# A CORRECT conversion -> LOW loss (~1-3) on coherent text.
#
#   IMAGE / MILES_HOST / HF_CKPT / LOAD_HOST as below.
#   TP=8 PP=2  (TP8xPP2 = 16 = world). DISABLE_OPT=1 for a load+fwd check only.
# Submit: cd <miles>; mkdir -p logs; sbatch examples/minimax_m3_vl/run_m3_real_2node.slurm
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-2/miles}
HF_CKPT=${HF_CKPT:-/fsx/peng/models/MiniMax-M3}
LOAD_HOST=${LOAD_HOST:-/fsx/peng/models/MiniMax-M3_torch_dist}
DATASETS_HOST=${DATASETS_HOST:-${MILES_HOST}/datasets}
DISABLE_OPT=${DISABLE_OPT:-1}   # 1 = load+forward/backward check (no optimizer state)
SAVE_HOST=${SAVE_HOST:-/fsx/peng/models/MiniMax-M3_sft_ckpt}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}   # >0 = save a (weights-only) torch_dist ckpt every N steps

MILES_ROOT=/workspace/miles; HF_ROOT=/root/m3_hf; LOAD_ROOT=/root/m3_load; DATASETS_ROOT=/root/datasets; SAVE_ROOT=/root/m3_save
TP=${TP:-8}; PP=${PP:-2}; EP=${EP:-1}; ETP=${ETP:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}

RAY_TMP_HOST=/tmp/ray-tmp-${USER:-peng}-${SLURM_JOB_ID:-manual}; RAY_TMP_ROOT=/tmp/ray-session
srun --overlap --ntasks-per-node=1 bash -c "mkdir -p ${RAY_TMP_HOST}"
[[ "${SAVE_INTERVAL}" != "0" ]] && mkdir -p "${SAVE_HOST}"
MOUNTS="${MILES_HOST}:${MILES_ROOT},${HF_CKPT}:${HF_ROOT},${LOAD_HOST}:${LOAD_ROOT},\
${DATASETS_HOST}:${DATASETS_ROOT},${RAY_TMP_HOST}:${RAY_TMP_ROOT},${SAVE_HOST}:${SAVE_ROOT},/var/log/aws/clusters:/var/log/aws/clusters"

# ---------- cluster discovery (2 nodes) ----------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${nodes[0]}
# NIC names differ per node on this cluster (AWS varies the PCI bus id, e.g.
# enp73s0 vs enp75s0). Pinning the head node's exact name makes NCCL bootstrap
# fail on any other node ("no socket interface found" -> ncclInvalidUsage).
# So detect the iface on EVERY node and use their common prefix (OPD template).
common_prefix() {
  local prefix=$1; shift; local value
  for value in "$@"; do
    while [[ -n "${prefix}" && "${value#"$prefix"}" == "${value}" ]]; do prefix=${prefix%?}; done
  done
  printf '%s' "${prefix}"
}
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
# EFA/RDMA NCCL settings this cluster needs for multinode collectives (from OPD template).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_PROTO="simple" NCCL_SOCKET_FAMILY="AF_INET"
export RDMAV_FORK_SAFE=1 FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${ALL_HOSTS_CSV}"
echo "== M3 real | head=${HEAD_NODE} (${MASTER_ADDR}) | nodes=${#nodes[@]} | TP${TP} PP${PP} EP${EP} =="

cleanup(){ local ec=$?; trap - EXIT INT TERM; set +e
  for n in "${nodes[@]}"; do srun --overlap --nodes=1 --ntasks=1 -w "$n" --container-image "${IMAGE}" \
    --container-mounts "${MOUNTS}" --container-remap-root bash -lc "ray stop --force||true"||true; done; exit "$ec"; }
trap cleanup EXIT INT TERM
PIDS=()

# ---------- ray head ----------
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
  bash -lc "set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
    # Pre-warm the trust_remote_code dynamic-module cache ONCE per node so the 8
    # actors don't race writing/importing transformers_modules.m3_hf (flaky
    # AttributeError: ... has no attribute 'MiniMaxM3VLConfig').
    python3 -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('${HF_ROOT}', trust_remote_code=True)\" >/dev/null 2>&1 || true
    ray stop --force||true
    ray start --head --node-ip-address=${MASTER_ADDR} --port=${MASTER_PORT} --num-gpus ${GPUS_PER_NODE} \
      --temp-dir=${RAY_TMP_ROOT} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block" &
PIDS+=($!); sleep 20

# ---------- ray workers ----------
for ((i=1;i<${#nodes[@]};i++)); do
  srun --nodes=1 --ntasks=1 -w "${nodes[$i]}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
    bash -lc "set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
      pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
      python3 -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('${HF_ROOT}', trust_remote_code=True)\" >/dev/null 2>&1 || true
      ray stop --force||true
      ray start --address ${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${GPUS_PER_NODE} --temp-dir=${RAY_TMP_ROOT} --disable-usage-stats --block" &
  PIDS+=($!); sleep 5
done
sleep 25
for a in $(seq 1 36); do timeout 2 bash -c "</dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null && break; sleep 5; done

# ---------- submit ----------
set +e
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --container-image "${IMAGE}" --container-mounts "${MOUNTS}" \
  --container-remap-root --container-env=NCCL_SOCKET_IFNAME,NCCL_DEBUG,no_proxy \
  bash -lc "
    set -euxo pipefail; cd ${MILES_ROOT}; export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1|tail -2
    source scripts/models/minimax-m3-428B.sh
    OPT=( --optimizer adam --lr ${LR:-1e-6} --lr-decay-style constant --weight-decay 0.0 )
    # 428B optimizer (fp32 master + Adam m/v) does NOT fit on GPU alongside params:
    # OOM at first optimizer.step() fused_adam._initialize_state. Offload it to CPU.
    # bf16 Adam moments (m,v): 12->8 B/param of optimizer state, keeps fp32 master.
    # Cuts host RAM ~1.9TB->~1.45TB/node so 4 nodes (EP32) fits. EP=32 needs world%32==0
    # => only 4 or 8 nodes valid (6 nodes forces EP<=16 -> GPU-OOM). See run notes.
    [[ '${CPU_OFFLOAD:-1}' == '1' ]] && OPT+=( --optimizer-cpu-offload --optimizer-offload-fraction 1.0 --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 )
    [[ '${DISABLE_OPT}' == '1' ]] && OPT=( --debug-disable-optimizer )
    # Save a weights-only torch_dist checkpoint to test the SAVE path (round-trips with --load).
    SAVE_FLAGS=()
    [[ '${SAVE_INTERVAL}' != '0' ]] && SAVE_FLAGS=( --save ${SAVE_ROOT} --save-interval ${SAVE_INTERVAL} --no-save-optim )
    ray job submit --address=http://${MASTER_ADDR}:8265 \
      --runtime-env-json='{\"env_vars\":{\"PYTHONPATH\":\"${MILES_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"RAY_DEDUP_LOGS\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"NCCL_SOCKET_IFNAME\":\"${IFACE}\",\"NCCL_PROTO\":\"simple\",\"NCCL_SOCKET_FAMILY\":\"AF_INET\",\"RDMAV_FORK_SAFE\":\"1\",\"FI_EFA_USE_DEVICE_RDMA\":\"1\",\"FI_EFA_SET_CUDA_SYNC_MEMOPS\":\"0\",\"TORCH_NCCL_ASYNC_ERROR_HANDLING\":\"1\",\"NCCL_DEBUG\":\"WARN\",\"no_proxy\":\"${no_proxy}\"}}' \
      -- python3 train.py --actor-num-nodes ${#nodes[@]} --actor-num-gpus-per-node ${GPUS_PER_NODE} \
      \"\${MODEL_ARGS[@]}\" \
      --hf-checkpoint ${HF_ROOT} --load ${LOAD_ROOT} \
      --prompt-data ${DATASETS_ROOT}/m3_smoke_text.parquet --input-key messages \
      --rollout-function-path miles.rollout.sft_rollout.generate_rollout --rollout-shuffle \
      --num-rollout ${NUM_ROLLOUT:-2} --rollout-batch-size 8 --global-batch-size 8 \
      --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only \
      --tensor-model-parallel-size ${TP} --sequence-parallel --pipeline-model-parallel-size ${PP} \
      --context-parallel-size 1 --expert-model-parallel-size ${EP} --expert-tensor-parallel-size ${ETP} \
      --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
      --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
      --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 \
      --attention-softmax-in-fp32 --attention-backend flash \
      \"\${OPT[@]}\" \"\${SAVE_FLAGS[@]}\"
  " 2>&1 | tee ./logs/m3-real-${SLURM_JOB_ID}.log
ec=${PIPESTATUS[0]}; set -e
[[ "${ec}" -ne 0 ]] && { echo "M3 real FAILED (exit ${ec})"; exit "${ec}"; }
echo "M3 real-weight run completed."
