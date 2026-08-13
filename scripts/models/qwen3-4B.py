import os


def model_args(rotary_base: str | None = None) -> str:
    rotary_base = rotary_base if rotary_base is not None else os.environ.get("MODEL_ARGS_ROTARY_BASE") or "1000000"
    return (
        "--swiglu "
        "--num-layers 36 "
        "--hidden-size 2560 "
        "--ffn-hidden-size 9728 "
        "--num-attention-heads 32 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        f"--rotary-base {rotary_base} "
        "--vocab-size 151936 "
        "--kv-channels 128 "
        "--qk-layernorm "
    )
