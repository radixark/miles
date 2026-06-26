NLAYERS="${MODEL_ARGS_NUM_LAYERS:-40}"
FIRST_K_DENSE_REPLACE=1

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers $NLAYERS
    --hidden-size 2048
    --ffn-hidden-size 7168
    --num-attention-heads 32
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280

    --multi-latent-attention
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-base "${MODEL_ARGS_ROTARY_BASE:-32000000}"
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    --num-experts 256
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size 768
    --moe-router-topk 8
    --moe-shared-expert-intermediate-size 768
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-permute-fusion
)
