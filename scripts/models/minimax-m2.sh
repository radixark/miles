MODEL_ARGS=(
    # Spec
    --spec "miles_plugins.models.minimax_m2" "get_minimax_m2_spec"

    # Numbers :: Common
    --num-layers 62
    --hidden-size 3072
    --vocab-size 200064

    # Numbers :: Attention
    --group-query-attention
    --num-attention-heads 48
    --num-query-groups 8
    --kv-channels 128

    # Numbers :: MoE
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 1536
    --moe-layer-freq [1]*62

    # Misc
    --disable-bias-linear
    --untie-embeddings-and-output-weights

    # Attention
    --qk-layernorm

    # MoE :: Router
    --moe-router-load-balancing-type seq_aux_loss  # may not be needed?
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0

    # MoE :: Primary
    --moe-grouped-gemm
    --moe-permute-fusion

    # Components :: Activation
    --swiglu

    # Components :: RMSNorm
    --normalization RMSNorm

    # Components :: RoPE
    --position-embedding-type rope
    --rotary-percent 0.5
    --rotary-base 5000000
)