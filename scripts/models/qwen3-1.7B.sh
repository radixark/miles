MODEL_ARGS=(
   --num-layers 28
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --rotary-base 1000000
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)