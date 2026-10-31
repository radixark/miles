import torch
import torch.nn as nn


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Group ``g`` of ``x`` [b, s, g, d] times row block ``g`` of ``weight`` [g * r, d]."""
    return torch.einsum("bsgd,grd->bsgr", x, weight.view(x.shape[2], -1, weight.shape[-1]))


class GroupedMatmul(nn.Module):
    """``grouped_matmul`` as a module, so a LoRA adapter can wrap the product."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return grouped_matmul(x, weight)
