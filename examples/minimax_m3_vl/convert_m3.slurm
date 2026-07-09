#!/bin/bash
#SBATCH --job-name=m3-convert
#SBATCH --nodes=1
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/m3-convert-%j.out
#SBATCH --error=./logs/m3-convert-%j.err
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --exclude=ip-10-1-8-7,ip-10-1-7-180,ip-10-1-38-198,ip-10-1-142-67,ip-10-1-126-226,ip-10-1-141-192,ip-10-1-94-38

# =========================================================================
# MiniMax-M3 (428B) HF -> Megatron torch_dist conversion.
#
# tools/convert_hf_to_torch_dist.py: builds the FULL 60-layer MSA model via
# --spec get_minimax_m3_spec, then mbridge (miles_plugins/mbridge/minimax_m3.py)
# maps HF safetensors -> Megatron params and saves a torch_dist checkpoint.
# Train afterwards with `--load <SAVE>` instead of `--debug-skip-load`.
#
# 428B bf16 ~= 856GB; sharded TP8/EP8 on 1 node (8xH200=1120GB) should fit (no
# optimizer states during conversion). Bump to 2 nodes (PP2) if it OOMs.
#
#   IMAGE=/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh
#   HF_CKPT=/fsx/peng/models/MiniMax-M3   SAVE=/fsx/peng/models/MiniMax-M3_torch_dist
#   TP=8 PP=1 EP=8
# Submit: cd <miles>; mkdir -p logs; sbatch examples/minimax_m3_vl/convert_m3.slurm
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-2/miles}
HF_CKPT=${HF_CKPT:-/fsx/peng/models/MiniMax-M3}
SAVE=${SAVE:-/fsx/peng/models/MiniMax-M3_torch_dist}

MILES_ROOT=/workspace/miles
HF_ROOT=/root/m3_hf
SAVE_ROOT=/root/m3_torch_dist
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
TP=${TP:-1}; PP=${PP:-1}; EP=${EP:-1}   # converter auto-sets PP=world_size when PP=1

mkdir -p "${SAVE}"
MOUNTS="${MILES_HOST}:${MILES_ROOT},${HF_CKPT}:${HF_ROOT},${SAVE}:${SAVE_ROOT},/var/log/aws/clusters:/var/log/aws/clusters"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export CUDA_DEVICE_MAX_CONNECTIONS=1

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3
    source scripts/models/minimax-m3-428B.sh   # -> MODEL_ARGS (incl. --spec)
    # The converter uses megatron's plain arg parser; drop miles-only flags it
    # does not define (e.g. --allgather-cp), else argparse exits 2.
    CONVERT_ARGS=()
    for a in \"\${MODEL_ARGS[@]}\"; do
      [[ \"\$a\" == \"--allgather-cp\" ]] && continue
      CONVERT_ARGS+=(\"\$a\")
    done
    PYTHONPATH=${MILES_ROOT}:/root/Megatron-LM/ \
    torchrun --nproc-per-node ${GPUS_PER_NODE} \
      tools/convert_hf_to_torch_dist.py \
      \"\${CONVERT_ARGS[@]}\" \
      --tensor-model-parallel-size ${TP} \
      --pipeline-model-parallel-size ${PP} \
      --expert-model-parallel-size ${EP} \
      --expert-tensor-parallel-size 1 \
      --hf-checkpoint ${HF_ROOT} \
      --save ${SAVE_ROOT} \
      --bf16
  "
echo "M3 conversion done -> ${SAVE}"
