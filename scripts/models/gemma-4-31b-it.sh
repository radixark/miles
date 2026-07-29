# Google Gemma-4 31B-it (BF16, DENSE — no experts).

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 16
   --kv-channels 256
   --num-layers 60
   --hidden-size 5376
   --ffn-hidden-size 21504
   --normalization RMSNorm
   --norm-epsilon 1e-06
   --position-embedding-type rope
   --rotary-base 1000000
   --vocab-size 262144
   --make-vocab-size-divisible-by 128
   --max-position-embeddings 262144
   # tied embeddings: do not pass --untie-embeddings-and-output-weights
)
