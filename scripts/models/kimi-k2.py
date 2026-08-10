from model_args_utils import moe_layer_freq


NLAYERS = 61
FIRST_K_DENSE_REPLACE = 1


def model_args() -> str:
    return (
        "--disable-bias-linear "
        "--num-layers 61 "
        "--hidden-size 7168 "
        "--ffn-hidden-size 18432 "
        "--num-attention-heads 64 "
        "--kv-channels 64 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 163840 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 128 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 128 "
        "--qk-layernorm "
        "--rotary-scaling-factor 32.0 "
        "--rotary-base 50000 "
        "--mscale 1.0 "
        "--mscale-all-dim 1.0 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        # moe
        "--num-experts 384 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=NLAYERS, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--moe-ffn-hidden-size 2048 "
        "--moe-router-topk 8 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-group-topk 1 "
        "--moe-router-num-groups 1 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 2.827 "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
    )
