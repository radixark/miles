# MiniMax-M3 (428B total / ~23B active) — MSA + MoE.
# Source from a run script the same way as scripts/models/glm5-744B-A40B.sh.
#
# Config: MiniMaxAI/MiniMax-M3 config.json (text backbone).
#   60 layers (0-2 dense full-attn, 3-59 MoE + MSA sparse attn),
#   hidden 6144, 64 q-heads / 4 kv-heads, head_dim 128, partial RoPE 64,
#   128 experts top-4 + 1 shared, sigmoid router + bias, routed_scaling 2.0.

MOE_ROUTED_EXPERTS=128
MOE_ACTIVE_ROUTED_EXPERTS=4
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=3072
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288          # dense layers 0-2 intermediate (dense_intermediate_size)
N_DENSE_LAYERS=3
N_MOE_LAYERS=57
NHEADS=64
NKVHEADS=4
HEAD_DIM=128

MODEL_ARGS=(
   --spec "miles_plugins.models.minimax_m3.minimax_m3" "get_minimax_m3_spec"
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$N_MOE_LAYERS
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.0     # routed_scaling_factor
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --make-vocab-size-divisible-by 16

    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --group-query-attention
    --num-query-groups $NKVHEADS
    --kv-channels $HEAD_DIM
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --rotary-percent 0.5                      # rotary_dim 64 / head_dim 128
    --rotary-base 5000000
    # Long-context (>256k) only: M3 uses YaRN (rope_type=yarn, factor 2.0,
    # original_max_position 1048576). Plain rope above is correct for typical
    # SFT/RL context; add `--use-rope-scaling` when training at ~1M ctx.
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm                            # per_head QK-norm
    --vocab-size 200064
    --enable-experimental

    # miles specific
    --allgather-cp
)
