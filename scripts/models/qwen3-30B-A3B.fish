set NLAYERS 48
set FIRST_K_DENSE_REPLACE 0

set arr
for i in (seq 0 (math $NLAYERS - 1))
    if test $i -lt $FIRST_K_DENSE_REPLACE
        set -a arr 0
    else
        set -a arr 1
    end
end

set MOE_LAYER_FREQ "["(string join ', ' $arr)"]"

if not set -q MODEL_ARGS_ROTARY_BASE
    set MODEL_ARGS_ROTARY_BASE 10000000
end

set MODEL_ARGS \
    --disable-bias-linear \
    --qk-layernorm \
    --group-query-attention \
    --num-attention-heads 32 \
    --num-query-groups 4 \
    --kv-channels 128 \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --rotary-percent 1.0 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --vocab-size 151936 \
    \
    --rotary-base $MODEL_ARGS_ROTARY_BASE \
    \
    --moe-ffn-hidden-size 768 \
    --moe-router-score-function softmax \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk 8 \
    --moe-layer-freq $MOE_LAYER_FREQ \
    --num-experts 128 \
    --moe-grouped-gemm \
    --moe-token-drop-policy probs \
    --moe-router-dtype fp32 \
    --moe-permute-fusion \
    --moe-aux-loss-coeff 0
