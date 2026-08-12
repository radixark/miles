import os

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16


def nvfp4_weight_e4m3_max() -> int:
    if os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all") and os.getenv(
        "NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"
    ).strip().lower() in ("weights", "all"):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    fp4_max = torch.tensor(FP4_E2M1_MAX, device=global_amax.device, dtype=torch.float32)
    fp8_max = torch.tensor(float(nvfp4_e4m3_max), device=global_amax.device, dtype=torch.float32)
    global_encode_scale = torch.div(fp8_max * fp4_max, global_amax.to(torch.float32))
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale.numel() == 1:
        if global_encode_scale == torch.tensor(0.0, device=global_amax.device, dtype=torch.float32):
            global_encode_scale = torch.tensor(1.0, device=global_amax.device, dtype=torch.float32)
    else:
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
    return global_encode_scale


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max))


def _nvfp4_4over6_enabled() -> bool:
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all")


def _pad_rows_for_te_quantizer(weight: torch.Tensor) -> torch.Tensor:
    pad_rows = (-weight.shape[0]) % TE_NVFP4_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


E4M3_MIN_SUBNORMAL = 2.0**-9
# Midpoints between adjacent E2M1 magnitudes, for nearest-value encoding.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _reencode_zero_scale_blocks(
    weight: torch.Tensor,
    qweight: torch.Tensor,
    block_scale: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TE's NVFP4 kernel flushes a block to zero when its E4M3 scale
    underflows; ModelOpt clamps the block scale at the smallest E4M3
    subnormal instead. Re-encode flushed blocks on the clamped grid so tiny
    weights survive (and ModelOpt checkpoints round-trip byte-for-byte)."""
    zero_blocks = block_scale.float() == 0
    if not zero_blocks.any():
        return qweight, block_scale

    norm = weight.float() / (E4M3_MIN_SUBNORMAL * global_decode_scale.float())
    midpoints = torch.tensor(_E2M1_MIDPOINTS, device=weight.device, dtype=torch.float32)
    mag_idx = torch.bucketize(norm.abs().contiguous(), midpoints).clamp(max=7)
    nibble = (mag_idx + torch.signbit(norm) * 8).to(torch.uint8)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)

    byte_mask = zero_blocks.repeat_interleave(NVFP4_GROUP_SIZE // 2, dim=-1)
    qweight = torch.where(byte_mask, packed, qweight)
    block_scale = torch.where(
        zero_blocks,
        torch.tensor(E4M3_MIN_SUBNORMAL, device=block_scale.device),
        block_scale.float(),
    ).to(torch.float8_e4m3fn)
    return qweight, block_scale


def nvfp4_quantize_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=_nvfp4_4over6_enabled(),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        with_random_sign_mask=False,
    )

    quantized = quantizer.quantize(_pad_rows_for_te_quantizer(weight))
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[:num_rows, : num_cols // NVFP4_GROUP_SIZE].contiguous()
    amax = quantized._amax_rowwise.reshape(-1)[0]
    global_decode_scale = nvfp4_global_decode_scale_te(amax, nvfp4_e4m3_max)
    qweight, block_scale = _reencode_zero_scale_blocks(
        weight, qweight, block_scale.view(torch.float8_e4m3fn), global_decode_scale
    )
    return qweight, block_scale, global_decode_scale


def nvfp4_quantize_1d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError("nvfp4_quantize_1d_pair expects two 2D tensors.")
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"NVFP4 paired quantization requires matching K dimensions, got {first.shape[1]} and {second.shape[1]}."
        )

    first_rows = first.shape[0]
    combined = _contiguous_pair_view(first, second)
    if combined is None:
        combined = torch.cat((first.contiguous(), second.contiguous()), dim=0)
    combined_qweight, combined_block_scale, global_scale = nvfp4_quantize_1d(combined)
    first_result = (
        combined_qweight[:first_rows].contiguous(),
        combined_block_scale[:first_rows].contiguous(),
        global_scale.clone(),
    )
    second_result = (
        combined_qweight[first_rows:].contiguous(),
        combined_block_scale[first_rows:].contiguous(),
        global_scale.clone(),
    )
    return first_result, second_result


def _contiguous_pair_view(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor | None:
    if not first.is_contiguous() or not second.is_contiguous():
        return None
    if first.device != second.device or first.dtype != second.dtype or first.stride() != second.stride():
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if first.storage_offset() + first.numel() != second.storage_offset():
        return None

    try:
        return first.as_strided(
            (first.shape[0] + second.shape[0], first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )
    except RuntimeError:
        return None
