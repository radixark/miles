MODEL_ARGS=(
  --swiglu
  --num-layers 28
  --hidden-size 3584
  --ffn-hidden-size 18944
  --num-attention-heads 28
  --use-rotary-position-embeddings
  --disable-bias-linear
  --add-qkv-bias
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --rotary-base 10000
  --group-query-attention
  --num-query-groups 4
  --vocab-size 152064
  --untie-embeddings-and-output-weights
)
