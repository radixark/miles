"""The padded lengths of a fixed-shape (TPU) torchtitan microbatch, kept torch-free for the rollout side."""

_MIN_FIXED_SHAPE_BUCKET = 1024


def fixed_shape_length_bucket(n_tokens: int, seq_len: int) -> int:
    return min(max(_MIN_FIXED_SHAPE_BUCKET, 1 << (n_tokens - 1).bit_length()), seq_len)


def fixed_shape_length_buckets(seq_len: int) -> set[int]:
    return {fixed_shape_length_bucket(1 << k, seq_len) for k in range(seq_len.bit_length() + 1)}
