"""MXFP4 E2M1 quantization utilities."""

import torch


MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
E2M1_MAX = 6.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Magnitudes an E2M1 element can take, indexed by its three low bits. The
# midpoints between them decide which code a value rounds to.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def mxfp4_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to packed MXFP4 with one UE8M0 scale per 32 elements."""
    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP4_GROUP_SIZE} for MXFP4.")

    blocks = weight.float().reshape(-1, MXFP4_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / E2M1_MAX).clamp(min=-float(E8M0_BIAS)))
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -float(E8M0_BIAS)))

    scaled = blocks / torch.exp2(exponent)
    magnitude = (scaled.abs().unsqueeze(-1) > _E2M1_BOUNDS.to(weight.device)).sum(dim=-1)
    codes = (magnitude + torch.signbit(scaled).to(magnitude.dtype) * 0b1000).to(torch.uint8)

    codes = codes.reshape(*weight.shape[:-1], k)
    packed = (codes[..., 1::2] << 4) | codes[..., 0::2]
    scale = (exponent + E8M0_BIAS).to(torch.uint8).reshape(*weight.shape[:-1], k // MXFP4_GROUP_SIZE)
    return packed.view(torch.int8), scale.view(torch.float8_e8m0fnu)
