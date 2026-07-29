import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
