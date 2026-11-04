import torch
import torch.nn as nn

from miles_plugins.models import cake_batched_projection


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Group ``g`` of ``x`` [b, s, g, d] times row block ``g`` of ``weight`` [g * r, d]."""
    return torch.einsum("bsgd,grd->bsgr", x, weight.view(x.shape[2], -1, weight.shape[-1]))


class GroupedMatmul(nn.Module):
    """``grouped_matmul`` as a module, so a LoRA adapter can wrap the product. ``backend`` 'cake'
    (--batched-projection-gemm-backend) runs it on the Cake deterministic GEMM."""

    def __init__(self, backend: str = "torch"):
        super().__init__()
        self.backend = backend

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.backend == "torch":
            return grouped_matmul(x, weight)
        return cake_batched_projection.grouped_matmul(x, weight, backend=self.backend)
