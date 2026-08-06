NLAYERS="${MODEL_ARGS_NUM_LAYERS:-42}"

# Inkling-Small 276B config (42 layers; derived from the HF config the same way
# inkling.sh maps the Inkling one).
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers $NLAYERS
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 201024
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --attention-softmax-in-fp32
    --position-embedding-type none
    --no-rope-fusion
    --no-masked-softmax-fusion
    --max-position-embeddings 1048576

    # MoE
    --num-experts 256
    --moe-ffn-hidden-size 2048
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2048
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --qk-layernorm

    # Inkling model provider
    --custom-model-provider-path miles_plugins.models.inkling.model.inkling_model_provider
)
