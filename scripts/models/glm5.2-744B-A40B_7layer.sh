SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/glm5.2-744B-A40B.sh"

# Override for the 7-layer pruned GLM-5.2 toy (jybsuper/GLM-5.2-7layer):
# first 7 layers = 3 dense + 4 MoE.  (Same pattern as glm5.1-744B-A40B_6layer.sh.)
#
# This keeps the DSA cross-layer index-sharing schedule exercised: with
# index_topk_freq=4 / index_skip_topk_offset=3, the computing layers are the
# Megatron 1-indexed 1,2,3,7 (they carry indexer weights in the checkpoint) and
# 4,5,6 are skip layers that reuse layer 3's top-k.
N_MOE_LAYERS=4

for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[$i]}" in
        --num-layers) MODEL_ARGS[$((i+1))]=$((N_DENSE_LAYERS + N_MOE_LAYERS)) ;;
        --moe-layer-freq) MODEL_ARGS[$((i+1))]="[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}" ;;
    esac
done

# NOTE on --spec (inherited from glm5.2-744B-A40B.sh): in the LoRA-via-bridge path
# (--megatron-to-hf-mode bridge + --lora-rank>0, e.g. scripts/run_glm5_lora.py) the model is
# built by the Megatron-Bridge GLM5 provider -- which reads index_topk_freq from the HF config
# and builds CrossLayerDSAttention -- NOT get_glm5_spec. So --spec is inert for the LoRA path;
# it is kept here only for parity with the full-FT / slime path that still consumes it.
