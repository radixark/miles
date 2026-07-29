# Google Gemma-4 26B-A4B-it (BF16, MoE: 128 experts, top-k 8).

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 8
   --kv-channels 256
   --num-layers 30
   --hidden-size 2816
   --ffn-hidden-size 2112
   --normalization RMSNorm
   --norm-epsilon 1e-06
   --position-embedding-type rope
   --rotary-base 1000000
   --vocab-size 262144
   --make-vocab-size-divisible-by 128
   --max-position-embeddings 262144
   # tied embeddings: do not pass --untie-embeddings-and-output-weights

   --num-experts 128
   --moe-router-topk 8
   --moe-ffn-hidden-size 704
   --moe-router-score-function softmax
   --moe-grouped-gemm
   --moe-router-dtype fp32
   --moe-router-num-groups 1
   --moe-router-group-topk 1
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-bias-update-rate 0
   --moe-aux-loss-coeff 0
)
