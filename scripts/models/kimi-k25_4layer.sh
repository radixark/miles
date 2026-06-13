SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/kimi-k2-thinking.sh"

# Override for the 4-layer pruned debugging model (first_k_dense_replace=1):
# 1 dense layer + 3 MoE layers. Architecture is otherwise identical to the full
# Kimi-K2.5 / K2-Thinking, so we reuse those MODEL_ARGS and only patch the
# layer count and the MoE-layer-frequency mask.
NLAYERS=4
FIRST_K_DENSE_REPLACE=1

arr=()
for ((i = 0; i < NLAYERS; i++)); do
    if ((i < FIRST_K_DENSE_REPLACE)); then
        arr+=(0)
    else
        arr+=(1)
    fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

for ((i = 0; i < ${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[$i]}" in
        --num-layers) MODEL_ARGS[$((i + 1))]=$NLAYERS ;;
        --moe-layer-freq) MODEL_ARGS[$((i + 1))]="$MOE_LAYER_FREQ" ;;
    esac
done
