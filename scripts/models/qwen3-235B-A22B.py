import os

from model_args_utils import moe_layer_freq


NLAYERS = 94
FIRST_K_DENSE_REPLACE = 0


def model_args(rotary_base: str | None = None) -> str:
    rotary_base = rotary_base if rotary_base is not None else os.environ.get("MODEL_ARGS_ROTARY_BASE") or "1000000"
    return (
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 64 "
        "--num-query-groups 4 "
        "--kv-channels 128 "
        "--num-layers 94 "
        "--hidden-size 4096 "
        "--ffn-hidden-size 12288 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 1.0 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 151936 "
        f"--rotary-base {rotary_base} "
        # moe
        "--moe-ffn-hidden-size 1536 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=NLAYERS, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 128 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
    )
