"""MXFP4 (E2M1 elements + E8M0 block scales) pack/unpack.

Torch-only, so checkpoint tooling can use it without importing Megatron.
"""

import torch

MXFP4_GROUP_SIZE = 32
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
# Block amaxes below this floor share the smallest normal power-of-two scale, 2**-126.
MXFP4_AMAX_FLOOR = E2M1_MAX * 2.0**-126


def _biased_exponent_of_ceil_pow2(value: torch.Tensor) -> torch.Tensor:
    """Biased fp32 exponent of 2 ** ceil(log2(value)) for positive normal fp32 values, exact at powers of two."""
    bits = value.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def quantize_mxfp4(weight, group_size):
    """Quantize to MXFP4: E2M1 elements rounded to nearest with ties to even, and E8M0 block scales.

    Each block's scale is the smallest power of two that keeps its amax within E2M1_MAX, computed as
    ``2 ** ceil(log2(amax * (1 / 6)))`` in fp32, so no element saturates. The arithmetic matches
    ``miles.kernels.quant.fake_quant.fake_quant_fp4_forward`` bit for bit, which lets fake
    quantization in training reproduce the weights a rollout decodes from this encoding.
    """
    assert weight.shape[-1] % group_size == 0
    assert weight.shape[-1] % 2 == 0

    blocks = weight.reshape(-1, group_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(MXFP4_AMAX_FLOOR)
    biased_exponent = _biased_exponent_of_ceil_pow2(amax * (1.0 / E2M1_MAX))
    normalized = blocks / (biased_exponent << 23).view(torch.float32)

    magnitude = normalized.abs().clamp_max(E2M1_MAX)
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    rounded = torch.round(magnitude / step) * step
    code = torch.where(rounded < 2.0, rounded * 2.0, torch.where(rounded < 4.0, rounded + 2.0, rounded * 0.5 + 4.0))
    encoded = code.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)

    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2).contiguous()
    scale = biased_exponent.to(torch.uint8)
    scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // group_size).contiguous()
    return packed, scale


def dequantize_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    assert weight_packed.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8
    assert group_size > 0

    unpacked = torch.empty(
        *weight_packed.shape[:-1],
        weight_packed.shape[-1] * 2,
        dtype=torch.uint8,
        device=weight_packed.device,
    )
    unpacked[..., 0::2] = weight_packed & 0x0F
    unpacked[..., 1::2] = (weight_packed >> 4) & 0x0F

    signs = 1.0 - 2.0 * ((unpacked & 0b1000) >> 3).float()
    magnitudes = unpacked & 0b0111
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=weight_packed.device)
    dequantized = signs * values[magnitudes.long()]

    assert dequantized.numel() % group_size == 0
    assert weight_scale.numel() == dequantized.numel() // group_size
    dequantized = dequantized.reshape(-1, group_size)
    scales = torch.exp2(weight_scale.float().reshape(-1, 1) - 127.0)
    return (dequantized * scales).reshape(unpacked.shape).to(torch.bfloat16).contiguous()


def mxfp4_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack weights with the MXFP4 parameter dtypes expected by SGLang."""
    if weight.shape[-1] % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {MXFP4_GROUP_SIZE} for MXFP4.")
    packed, scale = quantize_mxfp4(weight.contiguous(), group_size=MXFP4_GROUP_SIZE)
    return packed.view(torch.int8), scale.view(torch.float8_e8m0fnu)
