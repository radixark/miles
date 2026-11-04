"""MXFP4 (E2M1 elements + E8M0 block scales) pack/unpack.

Torch-only, so checkpoint tooling can use it without importing Megatron.
"""

import torch

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MXFP4_GROUP_SIZE = 32  # OCP MXFP4: one E8M0 scale per 32 elements


def quantize_mxfp4(weight, group_size):
    assert weight.shape[-1] % group_size == 0
    assert weight.shape[-1] % 2 == 0

    blocks = weight.reshape(-1, group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).float()
    scale_exp = torch.ceil(torch.log2(amax / 6.0)).clamp_(-127, 127)
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


def mxfp4_row_width(in_features: int, group_size: int) -> int:
    """Bytes per output row of a packed row tensor: K/2 codes, then K/group_size scales."""
    assert in_features % group_size == 0 and in_features % 2 == 0, (in_features, group_size)
    return in_features // 2 + in_features // group_size


def pack_mxfp4_rows(weight_packed: torch.Tensor, weight_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """[N, K/2] codes and [N, K/group_size] scales -> one [N, K/2 + K/group_size] uint8 tensor."""
    assert weight_packed.dtype == torch.uint8 and weight_scale.dtype == torch.uint8
    assert weight_packed.shape[0] == weight_scale.shape[0]
    assert weight_packed.shape[1] * 2 == weight_scale.shape[1] * group_size
    return torch.cat([weight_packed, weight_scale], dim=1)


def split_mxfp4_rows(rows: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes and scales a packed row tensor holds; the inverse of ``pack_mxfp4_rows``."""
    assert rows.dtype == torch.uint8
    width = rows.shape[-1]
    # width = K/2 + K/group_size = K (group_size + 2) / (2 group_size)
    assert (width * 2 * group_size) % (group_size + 2) == 0, (width, group_size)
    in_features = width * 2 * group_size // (group_size + 2)
    return rows[:, : in_features // 2], rows[:, in_features // 2 :]


def dequantize_mxfp4_rows(rows: torch.Tensor, group_size: int) -> torch.Tensor:
    """The BF16 [N, K] weight a packed row tensor encodes; exact."""
    return dequantize_mxfp4(*split_mxfp4_rows(rows, group_size), group_size)
