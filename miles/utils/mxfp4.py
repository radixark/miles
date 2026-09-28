"""MXFP4 (E2M1 elements + E8M0 block scales) pack/unpack, and how an FP8 checkpoint declares MXFP4 experts.

Torch-only, so checkpoint tooling can use it without importing Megatron.
"""

import torch

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _even_scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 block exponent: round amax to one mantissa bit, then floor(log2) minus the E2M1 emax (2).

    The scale steps up at amax = 7 * 2^s, halfway between the grid maxima 6 and 8, so no grid value sits on a
    boundary: a block whose max is 4 * 2^s or 6 * 2^s (every block of a checkpoint quantized this way or by
    floor(log2(amax)) - 2) keeps its scale while a trained weight drifts either way. ceil(log2(amax / 6)) doubles
    the scale on the first BF16 ULP of growth above 6 * 2^s, and plain floor halves it on the first ULP below
    4 * 2^s; either requantizes the block, so every weight sync moves the rollout experts away from the trained
    ones. Elements above 6 encode as 6. Same rule as torchao's ScaleCalculationMode.EVEN.
    """
    exponent = ((amax.contiguous().view(torch.int32) + 0x200000) & 0x7F800000) >> 23
    return (exponent - 127 - 2).clamp_(-127, 127).to(torch.float32)


def quantize_mxfp4(weight, group_size):
    assert weight.shape[-1] % group_size == 0
    assert weight.shape[-1] % 2 == 0

    blocks = weight.reshape(-1, group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).float()
    scale_exp = _even_scale_exponent(amax)
    normalized = blocks.float() * torch.exp2(-scale_exp)

    magnitude = torch.zeros_like(normalized, dtype=torch.uint8)
    for bound in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        magnitude.add_(normalized.abs() > bound)
    encoded = magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)

    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2).contiguous()
    scale = (scale_exp + 127).to(torch.uint8)
    scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // group_size).contiguous()
    return packed, scale


def has_mxfp4_routed_experts(quantization_config: dict) -> bool:
    """Whether an FP8 checkpoint declares MXFP4 routed experts (SGLang: ModelConfig.is_fp4_experts).

    `routed_experts_quant_method: mxfp4` is the generic key. `store_dtype: mxfp4` is MiMo-V2's own key for the
    same layout. DeepSeek-V4 declares neither; SGLang detects its FP4 experts from the weights.
    """
    return "mxfp4" in (quantization_config.get("routed_experts_quant_method"), quantization_config.get("store_dtype"))


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
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=weight_packed.device)
    dequantized = signs * values[magnitudes.long()]

    assert dequantized.numel() % group_size == 0
    assert weight_scale.numel() == dequantized.numel() // group_size
    dequantized = dequantized.reshape(-1, group_size)
    scales = torch.exp2(weight_scale.float().reshape(-1, 1) - 127.0)
    return (dequantized * scales).reshape(unpacked.shape).to(torch.bfloat16).contiguous()


def project_mxfp4(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """The BF16 values the MXFP4 encoding of ``weight`` decodes to, i.e. what an MXFP4 engine serves.

    Idempotent: a projected weight re-encodes to the same values, so a trainer that computes with projected
    weights hands the engine exactly what it trains.
    """
    return dequantize_mxfp4(*quantize_mxfp4(weight, group_size), group_size)
