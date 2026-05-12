NLAYERS=62
FIRST_K_DENSE_REPLACE=0

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
   --spec "miles_plugins.models.minimax_m2" "get_minimax_m2_spec"
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 48
   --num-query-groups 8
   --kv-channels 128
   --num-layers 62
   --hidden-size 3072
   --ffn-hidden-size 1536

   --normalization RMSNorm
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.5
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 200064

   --rotary-base 5000000

   # moe
   --moe-ffn-hidden-size 1536
   --moe-router-score-function sigmoid
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq $MOE_LAYER_FREQ
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-router-enable-expert-bias
   --moe-aux-loss-coeff 0
)
