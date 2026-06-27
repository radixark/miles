# MiniMax-M3 — REDUCED 4-layer smoke config (1 dense + 3 MoE+MSA layers).
# Random-init plumbing test ONLY (weights won't load — layer count differs from
# the real 60-layer checkpoint). Mirrors scripts/models/minimax-m3-428B.sh but
# tiny, so spec-build -> MSA forward -> (optional VL merge) -> loss runs on 1 node.

MOE_ROUTED_EXPERTS=128
MOE_ACTIVE_ROUTED_EXPERTS=4
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=3072
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
N_DENSE_LAYERS=1          # 1 dense (layer 0) so at least one MSA layer exists
N_MOE_LAYERS=3           # 3 MoE + MSA sparse layers (layers 1-3)
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
    --moe-router-topk-scaling-factor 2.0
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
    --rotary-percent 0.5
    --rotary-base 5000000
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
    --vocab-size 200064
    --enable-experimental

    # miles specific
    --allgather-cp
)
