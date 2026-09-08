import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def fixed_tree_mean_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Mean over the last dimension with a fixed binary reduction tree.

    This keeps the floating-point association independent of all outer tensor
    dimensions. The input must be a non-scalar floating-point tensor whose dtype
    supports addition and division and whose last dimension is a nonzero power
    of two. DeepSeek-V4 TOP uses this for the 512-wide query RMS in
    batch-invariant mode.
    """
    if x.ndim == 0:
        raise RuntimeError("fixed_tree_mean_last_dim requires a tensor with at least one dimension.")
    if not x.is_floating_point():
        raise RuntimeError(f"fixed_tree_mean_last_dim requires a floating-point tensor, got {x.dtype}.")

    width = x.shape[-1]
    if width == 0 or width & (width - 1):
        raise RuntimeError(f"fixed_tree_mean_last_dim requires a nonzero power-of-two last dimension, got {width}.")

    reduced = x
    while reduced.shape[-1] > 1:
        reduced = reduced.reshape(*reduced.shape[:-1], reduced.shape[-1] // 2, 2)
        reduced = reduced[..., 0] + reduced[..., 1]
    return reduced / width


def dsv4_query_rms_norm(
    x: torch.Tensor,
    eps: float,
    *,
    batch_invariant: bool,
) -> torch.Tensor:
    """Apply DSV4's unweighted 512-wide query-head RMS normalization.

    The batch-invariant branch is the trainer-side half of the query-RMS
    contract. Exact rollout/prefill parity also requires the matching SGLang
    path, which is intentionally deferred to a follow-up change.
    """
    x_fp32 = x.float()
    x_square = x_fp32.square()
    x_square_mean = fixed_tree_mean_last_dim(x_square) if batch_invariant else x_square.mean(-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(x_square_mean + eps)).to(x.dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
