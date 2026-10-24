import torch
import torch.nn as nn


class GroupedMatmul(nn.Module):
    """Group ``g`` of ``x`` times row block ``g`` of ``weight``; a module so a LoRA adapter can wrap the product."""

    def __init__(self, n_groups: int):
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsgd,grd->bsgr", x, weight.view(self.n_groups, -1, weight.shape[-1]))
