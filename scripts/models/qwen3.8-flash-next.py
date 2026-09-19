from model_args_utils import moe_layer_freq

FIRST_K_DENSE_REPLACE = 0


def model_args(nlayers: int = 48) -> str:
    """Qwen3.8-Flash-Next: 180B total, ~7.4B active. Shapes from the released
    config.json; the hyper-connection / PLE / QSA fields have no Megatron CLI
    flags and are derived from the checkpoint by the spec instead
    (miles_plugins/models/qwen3_8_next/qwen3_8_next.py). --mtp-num-layers is
    omitted: MTP tensors are not yet mapped."""
    return (
        "--spec miles_plugins.models.qwen3_8_next.qwen3_8_next get_qwen3_8_next_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 24 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        f"--num-layers {nlayers} "
        "--hidden-size 2560 "
        "--ffn-hidden-size 640 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        "--moe-ffn-hidden-size 640 "
        "--moe-shared-expert-intermediate-size 640 "
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
        "--attention-output-gate "
        "--moe-shared-expert-gate "
    )
