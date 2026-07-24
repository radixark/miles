import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def fixed_tree_mean_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Mean over the last dimension with a fixed binary reduction tree.

    This keeps the floating-point association independent of all outer tensor
    dimensions while remaining composed of ordinary autograd-aware PyTorch
    operations. DeepSeek-V4 TOP currently uses it for the 512-wide query RMS.
    """
    width = x.shape[-1]
    if width == 0 or width & (width - 1):
        raise RuntimeError("fixed_tree_mean_last_dim requires a nonzero power-of-two last " f"dimension, got {width}.")

    reduced = x
    while reduced.shape[-1] > 1:
        reduced = reduced.reshape(*reduced.shape[:-1], reduced.shape[-1] // 2, 2)
        reduced = reduced[..., 0] + reduced[..., 1]
    return reduced / width


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
