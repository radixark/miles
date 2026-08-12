from model_args_utils import moe_layer_freq


FIRST_K_DENSE_REPLACE = 0


def model_args(nlayers: int = 92, mtp_num_layers: int | None = 1) -> str:
    mtp = f"--mtp-num-layers {mtp_num_layers} " if mtp_num_layers else ""
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 64 "
        "--num-query-groups 4 "
        "--kv-channels 256 "
        f"--num-layers {nlayers} "
        "--hidden-size 8192 "
        "--ffn-hidden-size 2048 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # moe
        "--moe-ffn-hidden-size 2048 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 10 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 512 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        # qwen3.5 architecture family
        "--attention-output-gate "
        "--moe-shared-expert-gate "
        f"{mtp}"
    )
