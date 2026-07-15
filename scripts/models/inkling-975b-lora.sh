source "$(dirname -- "${BASH_SOURCE[0]}")/inkling-975b.sh"

# Inkling 975B LoRA (r=32, all-linear): adapter-only training and weight sync;
# the engine serves the adapter natively (triton backend, virtual experts).
LORA_RANK="${LORA_RANK:-32}"

LORA_ARGS=(
    --lora-rank $LORA_RANK
    --lora-alpha $LORA_RANK
    --target-modules all-linear
    --sglang-lora-backend triton
    --sglang-lora-use-virtual-experts
    --sglang-max-loras-per-batch 1
    --sglang-max-lora-rank $LORA_RANK
)
