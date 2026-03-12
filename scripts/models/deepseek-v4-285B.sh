NLAYERS="${MODEL_ARGS_NUM_LAYERS:-43}"

# V4: all layers are MoE
arr=()
for ((i=0; i<NLAYERS; i++)); do
  arr+=(1)
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# compress_ratios from config_285B.json: [0, 0, 4, 128, 4, 128, ...]
if [ ${#COMPRESS_RATIOS[@]} -eq 0 ]; then
  COMPRESS_RATIOS=(0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4)
fi

# DeepSeek V4 285B config
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers $NLAYERS
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # MLA params (reused by V4)
    --multi-latent-attention
    --q-lora-rank 1024
    --kv-lora-rank 512
    --qk-head-dim 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --qk-layernorm
    --rotary-scaling-factor 4
    --rotary-base 10000
    --original-max-position-embeddings 65536
    --beta-fast 32
    --beta-slow 1
    --attention-softmax-in-fp32
    --no-rope-fusion

    # MoE
    --num-experts 256
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size 2048
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2048
    --moe-router-pre-softmax
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 1.5

    # DSV4 specific
    --experimental-attention-variant dsv4
    --dsv4-hc-mult 4
    --dsv4-hc-sinkhorn-iters 20
    --dsv4-compress-ratios "${COMPRESS_RATIOS[@]}"
    --dsv4-compress-rope-theta 160000
    --dsv4-o-groups 8
    --dsv4-o-lora-rank 1024
    --dsv4-n-hash-layers 3
    --dsv4-window-size 128

    # DSA Indexer
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512

    # V4 model spec (plugin)
    --spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec
)
