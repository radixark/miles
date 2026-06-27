#!/bin/bash
#SBATCH --job-name=m3-to-hf
#SBATCH --nodes=1
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/m3-to-hf-%j.out
#SBATCH --error=./logs/m3-to-hf-%j.err
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --exclude=ip-10-1-142-67

# =========================================================================
# MiniMax-M3 (428B) Megatron torch_dist -> HF conversion (export a TRAINED ckpt).
#
# tools/convert_torch_dist_to_hf.py is SINGLE-PROCESS (no_dist=True): it reads the
# full DCP checkpoint into CPU RAM (~856GB bf16 for 428B) and writes safetensors.
# No GPU compute; we take a whole node just for its ~2TB RAM. Dispatch goes to
# miles/backends/megatron_utils/megatron_to_hf/minimax_m3.py (--model-name minimax_m3).
#
#   INPUT_HOST = saved iter dir from training (--save .../iter_000000N)
#   OUTPUT_HOST = HF safetensors output   ORIGIN_HF = config/tokenizer source
# Submit: cd <miles>; mkdir -p logs; sbatch examples/minimax_m3_vl/convert_torch_dist_to_hf_m3.slurm
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-2/miles}
INPUT_HOST=${INPUT_HOST:-/fsx/peng/models/MiniMax-M3_sft_ckpt/iter_0000005}
OUTPUT_HOST=${OUTPUT_HOST:-/fsx/peng/models/MiniMax-M3_sft_hf}
ORIGIN_HF=${ORIGIN_HF:-/fsx/peng/models/MiniMax-M3}
VOCAB_SIZE=${VOCAB_SIZE:-200064}

MILES_ROOT=/workspace/miles; INPUT_ROOT=/root/m3_in; OUTPUT_ROOT=/root/m3_out; ORIGIN_ROOT=/root/m3_origin
mkdir -p "${OUTPUT_HOST}"
MOUNTS="${MILES_HOST}:${MILES_ROOT},${INPUT_HOST}:${INPUT_ROOT},${OUTPUT_HOST}:${OUTPUT_ROOT},\
${ORIGIN_HF}:${ORIGIN_ROOT},/var/log/aws/clusters:/var/log/aws/clusters"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MOUNTS}" --container-remap-root \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3
    PYTHONPATH=${MILES_ROOT}:/root/Megatron-LM/ \
    python3 tools/convert_torch_dist_to_hf.py \
      --input-dir ${INPUT_ROOT} \
      --output-dir ${OUTPUT_ROOT} \
      --model-name minimax_m3 \
      --origin-hf-dir ${ORIGIN_ROOT} \
      --vocab-size ${VOCAB_SIZE} \
      -f
  "
echo "M3 torch_dist -> HF done -> ${OUTPUT_HOST}"
