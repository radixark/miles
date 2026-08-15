# shellcheck shell=bash
# NVIDIA Nemotron-3-Nano-30B-A3B (BF16) Megatron model args for Miles.
#
# These MODEL_ARGS are consumed by run.sh and pass straight through to
# miles/train.py (which delegates model-structural flags to the Megatron-LM
# backend). They mirror miles/scripts/models/nemotron-3-nano-30b-a3b.sh — the
# authoritative in-repo model recipe for this checkpoint — so keep them in sync
# with that file.
#
# The checkpoint is a hybrid MoE "nemotron_h" model (Mamba + GQA attention +
# MoE, 128 routed experts / 6 per-token / 1 shared). The mixed-Mamba/MoE weight
# layout is handled by Miles' AutoBridge path + the NemotronHBridge MoE shim
# (see miles/backends/megatron_utils/__init__.py) at load time from
# --hf-checkpoint; only attention-side structural args belong in MODEL_ARGS.
#
# HF config (verified): num_hidden_layers=52  hidden_size=2688
#   num_attention_heads=32  num_key_value_heads=2  head_dim=128
#   intermediate_size=1856  moe_intermediate_size=1856  n_routed_experts=128
#   num_experts_per_tok=6  n_shared_experts=1
#   moe_shared_expert_intermediate_size=3712  tie_word_embeddings=false
MODEL_ARGS=(
    --disable-bias-linear
    --group-query-attention
    --num-attention-heads 32
    --num-query-groups 2
    --kv-channels 128
    --num-layers 52
    --hidden-size 2688
    --ffn-hidden-size 1856
    --normalization RMSNorm
    --position-embedding-type none
    --vocab-size 131072
    --make-vocab-size-divisible-by 128
    --untie-embeddings-and-output-weights

    # MoE specifics
    --num-experts 128
    --moe-router-topk 6
    --moe-ffn-hidden-size 1856
    --moe-shared-expert-intermediate-size 3712
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-grouped-gemm
    --moe-router-dtype fp32
    # Routing: config has n_group=1 (MoE groups), topk_group=1,
    # routed_scaling_factor=2.5. `n_groups=8` is Mamba groups — unrelated to MoE.
    # With n_group=1, group-limited routing is a no-op (single group of 128).
    --moe-router-num-groups 1
    --moe-router-group-topk 1
    --moe-router-topk-scaling-factor 2.5
    --moe-router-pre-softmax
    # Match glm4.7-flash (known-working MoE RL) settings more closely.
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-bias-update-rate 0
    --moe-aux-loss-coeff 0
)
