def model_args() -> str:
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 40 "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        "--num-layers 42 "
        "--hidden-size 3136 "
        "--ffn-hidden-size 12544 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--vocab-size 131072 "
        "--make-vocab-size-divisible-by 128 "
        "--untie-embeddings-and-output-weights "
    )
