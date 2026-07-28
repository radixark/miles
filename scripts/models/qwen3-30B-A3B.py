import os

from model_args_utils import moe_layer_freq


FIRST_K_DENSE_REPLACE = 0


def model_args(nlayers: int | None = None, rotary_base: str | None = None) -> str:
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 48)
    rotary_base = rotary_base if rotary_base is not None else os.environ.get("MODEL_ARGS_ROTARY_BASE") or "1000000"
    return (
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 32 "
        "--num-query-groups 4 "
        "--kv-channels 128 "
        f"--num-layers {nlayers} "
        "--hidden-size 2048 "
        "--ffn-hidden-size 6144 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 1.0 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 151936 "
        f"--rotary-base {rotary_base} "
        # moe
        "--moe-ffn-hidden-size 768 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 128 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
    )
