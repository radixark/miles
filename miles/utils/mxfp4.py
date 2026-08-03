import torch
import triton
import triton.language as tl


MXFP4_GROUP_SIZE = 32


@triton.jit
def _mxfp4_dequantize_kernel(
    packed_ptr,
    scale_ptr,
    output_ptr,
    packed_cols: tl.constexpr,
    logical_cols: tl.constexpr,
    scale_cols: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    rows = offsets // logical_cols
    cols = offsets % logical_cols

    packed_offsets = rows * packed_cols + cols // 2
    packed = tl.load(packed_ptr + packed_offsets, mask=mask, other=0).to(tl.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    encoded = tl.where((cols & 1) == 0, low, high)

    magnitude_code = encoded & 0x07
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(magnitude_code == 5, 3.0, tl.where(magnitude_code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    value = tl.where((encoded & 0x08) == 0, magnitude, -magnitude)

    value = tl.where(magnitude_code == 0, 0.0, value)

    scale_offsets = rows * scale_cols + cols // 32
    scale_bits = tl.load(scale_ptr + scale_offsets, mask=mask, other=0).to(tl.uint8)
    scale = tl.exp2(scale_bits.to(tl.float32) - 127.0)
    scale = tl.where(scale_bits == 255, float("nan"), scale)
    tl.store(output_ptr + offsets, value * scale, mask=mask)


def mxfp4_dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize rowwise MXFP4 weights with compact, unswizzled UE8M0 scales."""
    if weight.device.type != "cuda" or scale.device.type != "cuda":
        raise ValueError("MXFP4 dequantization requires CUDA tensors.")
    if weight.device != scale.device:
        raise ValueError(f"MXFP4 weight and scale must use the same device, got {weight.device} and {scale.device}.")
    if weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"MXFP4 packed weights must use int8 or uint8 storage, got {weight.dtype}.")
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError("MXFP4 dequantization expects 2D weight and scale tensors.")
    if weight.shape[0] != scale.shape[0]:
        raise ValueError(f"MXFP4 weight and scale row counts must match, got {weight.shape} and {scale.shape}.")

    logical_cols = weight.shape[1] * 2
    if logical_cols % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"MXFP4 logical K={logical_cols} must be divisible by {MXFP4_GROUP_SIZE}.")
    expected_scale_shape = (weight.shape[0], logical_cols // MXFP4_GROUP_SIZE)
    if scale.shape != expected_scale_shape:
        raise ValueError(f"Expected MXFP4 scale shape {expected_scale_shape}, got {tuple(scale.shape)}.")

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        raise RuntimeError("MXFP4 dequantization requires torch.float8_e8m0fnu support.")
    if scale.dtype not in (e8m0_dtype, torch.uint8):
        raise ValueError(f"MXFP4 scales must use float8_e8m0fnu or uint8 storage, got {scale.dtype}.")

    packed_weight = weight.contiguous().view(torch.uint8)
    scale_bits = scale.contiguous().view(torch.uint8)
    output = torch.empty((weight.shape[0], logical_cols), dtype=dtype, device=weight.device)
    numel = output.numel()
    grid = (triton.cdiv(numel, 256),)
    _mxfp4_dequantize_kernel[grid](
        packed_weight,
        scale_bits,
        output,
        packed_cols=weight.shape[1],
        logical_cols=logical_cols,
        scale_cols=scale.shape[1],
        numel=numel,
        BLOCK_SIZE=256,
    )
    return output
