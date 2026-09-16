MOE_ROUTED_EXPERTS = 64
MOE_ACTIVE_ROUTED_EXPERTS = 4
MOE_SHARED_EXPERTS = 1
NHIDDEN = 2048
MOE_FFN_HIDDEN = 1536
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS
FFN_HIDDEN = 10240
N_DENSE_LAYERS = 1
N_MOE_LAYERS = 46
NHEADS = 20


def model_args() -> str:
    return (
        f"--moe-layer-freq [0]*{N_DENSE_LAYERS}+[1]*{N_MOE_LAYERS} "
        f"--num-experts {MOE_ROUTED_EXPERTS} "
        f"--moe-shared-expert-intermediate-size {MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} "
        f"--moe-router-topk {MOE_ACTIVE_ROUTED_EXPERTS} "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
        f"--moe-ffn-hidden-size {MOE_FFN_HIDDEN} "
        "--moe-router-score-function sigmoid "
        "--moe-router-pre-softmax "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-topk-scaling-factor 1.8 "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-dtype fp32 "
        "--make-vocab-size-divisible-by 64 "
        f"--num-layers {N_DENSE_LAYERS + N_MOE_LAYERS} "
        f"--hidden-size {NHIDDEN} "
        f"--ffn-hidden-size {FFN_HIDDEN} "
        f"--num-attention-heads {NHEADS} "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--position-embedding-type rope "
        "--rope-type rope "
        "--no-position-embedding "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--qk-layernorm "
        "--multi-latent-attention "
        "--q-lora-rank 768 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 192 "
        "--v-head-dim 256 "
        "--kv-channels 192 "
        "--qk-pos-emb-head-dim 64 "
        "--vocab-size 154880 "
        "--rotary-base 1000000 "
        "--no-rope-fusion "
        "--mtp-num-layers 1 "
    )
