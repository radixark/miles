SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/glm5-744B-A40B.sh"

# Override for the 6-layer pruned GLM-5.1 toy (jybsuper/GLM-5.1-6layer):
# first 6 layers = 3 dense + 3 MoE.  (Same pattern as glm5-744B-A40B_4layer.sh.)
N_MOE_LAYERS=3

for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[$i]}" in
        --num-layers) MODEL_ARGS[$((i+1))]=$((N_DENSE_LAYERS + N_MOE_LAYERS)) ;;
        --moe-layer-freq) MODEL_ARGS[$((i+1))]="[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}" ;;
    esac
done

# NOTE on --spec (inherited from glm5-744B-A40B.sh): in the LoRA-via-bridge path
# (--megatron-to-hf-mode bridge + --lora-rank>0) the model is built by the Megatron-Bridge
# provider + miles' "dsa" experimental-attention-variant monkey-patch (bridge_lora_helpers.py),
# NOT get_glm5_spec — args.spec is never imported/invoked there (model.py dispatch bypasses
# get_model_provider_func). So --spec is inert for the LoRA examples; it is kept here only for
# parity with the other glm5 registry entries (non-bridge / full-spec paths still use it).
