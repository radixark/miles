#!/bin/bash
# =============================================================================
# Convert a HF Qwen3.5-35B-A3B checkpoint -> Megatron torch_dist.
# Used to stage both the base (--ref-load) and the teacher (--opd-teacher-load)
# for phase2_gb200.sh, since neither is pre-staged on /cluster_public.
#
# Usage: convert_gb200.sh <hf_checkpoint_dir> <torch_dist_save_dir>
# =============================================================================
set -ex
HF_IN=${1:?hf checkpoint dir}
SAVE_OUT=${2:?torch_dist save dir}
MILES_DIR=${MILES_DIR:-/workspace/miles}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Identical architecture spec to phase2_gb200.sh's MODEL_ARGS.
MODEL_ARGS=(
   --spec miles_plugins.models.qwen3_5 get_qwen3_5_spec
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 16 --num-query-groups 2 --kv-channels 256
   --num-layers 40 --hidden-size 2048 --ffn-hidden-size 512
   --normalization RMSNorm --apply-layernorm-1p --position-embedding-type rope
   --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu
   --untie-embeddings-and-output-weights --vocab-size 248320 --rotary-base 10000000
   --moe-ffn-hidden-size 512 --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
   --num-experts 256 --moe-grouped-gemm --moe-token-drop-policy probs --moe-router-dtype fp32
   --moe-permute-fusion --moe-aux-loss-coeff 0 --attention-output-gate --moe-shared-expert-gate
   --mtp-num-layers 1
)

cd "${MILES_DIR}"
PYTHONPATH="${MILES_DIR}:${MEGATRON_PATH}" python3 "${MILES_DIR}/tools/convert_hf_to_torch_dist.py" \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_IN}" \
   --save "${SAVE_OUT}"
echo "CONVERTED ${HF_IN} -> ${SAVE_OUT}"
ls -la "${SAVE_OUT}"; cat "${SAVE_OUT}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "(no latest_checkpointed_iteration.txt yet)"
