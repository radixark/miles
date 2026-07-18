NLAYERS=93
MOE_LAYER_FREQ="[0,$(printf '1,%.0s' {1..91})1]"

MODEL_ARGS=(
    --spec "miles_plugins.models.kimi_k3" "get_kimi_k3_spec"

    --disable-bias-linear
    --num-layers "$NLAYERS"
    --hidden-size 7168
    --ffn-hidden-size 33792
    --num-attention-heads 96
    --num-query-groups 96
    --kv-channels 256
    --normalization RMSNorm
    --position-embedding-type none
    --rotary-base 10000
    --norm-epsilon 1e-5
    --hidden-dropout 0
    --attention-dropout 0
    --disable-bf16-reduced-precision-matmul
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 163840
    --make-vocab-size-divisible-by 128

    --multi-latent-attention
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --attention-softmax-in-fp32

    --num-experts 896
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size 3072
    --moe-latent-size 3584
    --moe-shared-expert-intermediate-size 6144
    --moe-router-topk 16
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --moe-router-bias-update-rate 0
    --moe-aux-loss-coeff 0
    --moe-router-topk-scaling-factor 1.0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
)
