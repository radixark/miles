import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Kept in pure PyTorch (FP32 weight + FP32 forward compute) rather than
    :class:`TENorm` because the Compressor runs its whole pipeline in FP32
    and explicitly requires it for numerical stability of the compressed-KV
    variance accumulation.

    Args:
        dim: Dimension of the input tensor.
        eps: Epsilon for numerical stability. Defaults to ``1e-6``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)
