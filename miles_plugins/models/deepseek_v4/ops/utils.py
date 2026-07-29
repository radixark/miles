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


def batch_of_row(cu_seqlens: torch.Tensor, total_rows: int) -> torch.Tensor:
    """Segment index owning each row of a THD-packed tensor.

    Args:
        cu_seqlens: ``[n_seg + 1]`` cumulative lengths.
        total_rows: number of rows; rows past ``cu_seqlens[-1]`` clamp to the last segment.
    Returns:
        ``[total_rows]`` int64.
    """
    n_seg = cu_seqlens.size(0) - 1
    row_idx = torch.arange(total_rows, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))
