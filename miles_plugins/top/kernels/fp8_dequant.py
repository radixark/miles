"""Stands in for SGLang's block_quant_dequant, which reads only FP32 block scales and materializes FP32 [n, k] scale
and product temporaries; this one also decodes serving's packed UE8M0 words, in the same pass, bit for bit
fp8_forward.dequantize_processed's eager formula.

Each 128x128 block's FP32 scale is decoded once per tile (from serving's packed UE8M0 word, as Hopper's unpack_ue8m0
decodes it, or read from FP32 block scales), the E4M3 payload widens exactly to FP32, the product is the same FP32
multiply eager torch does, and it rounds to BF16 once. Only the FP32 [n, k] temporaries go away.
"""

import torch
import triton
import triton.language as tl

_BLOCK_N = 64  # divides 128, so a tile never straddles a scale block
_BLOCK_K = 128


@triton.jit
def _dequant_kernel(
    payload,
    scale,
    out,
    n,
    k,
    tiles_k,
    p_matrix_stride,
    p_row_stride,
    s_matrix_stride,
    s_row_stride,
    s_col_stride,
    PACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    matrix = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(0)
    row0 = (tile // tiles_k) * BLOCK_N
    group = (tile % tiles_k) * BLOCK_K // 128
    if PACKED:
        # Serving's expanded packed form: the word for group g sits in row (r // 128) * 128, byte g % 4.
        word = tl.load(
            scale + matrix * s_matrix_stride + (row0 // 128 * 128) * s_row_stride + (group // 4) * s_col_stride
        )
        exponent = (word >> ((group % 4) * 8)) & 255
        bits = tl.where(exponent == 0, 0x00400000, tl.where(exponent == 255, 0x7FC00000, exponent << 23))
        block_scale = bits.to(tl.float32, bitcast=True)
    else:
        block_scale = tl.load(scale + matrix * s_matrix_stride + (row0 // 128) * s_row_stride + group * s_col_stride)
    rows = row0 + tl.arange(0, BLOCK_N)
    cols = group * 128 + tl.arange(0, BLOCK_K)
    live = (rows[:, None] < n) & (cols[None, :] < k)
    values = tl.load(
        payload + matrix * p_matrix_stride + rows[:, None].to(tl.int64) * p_row_stride + cols[None, :], mask=live
    )
    # Power-of-two scales times E4M3 values are exact in FP32; the BF16 rounding is the eager .to(torch.bfloat16).
    decoded = (values.to(tl.float32) * block_scale).to(tl.bfloat16)
    tl.store(out + matrix * n * k + rows[:, None].to(tl.int64) * k + cols[None, :], decoded, mask=live)


def dequantize_blocks(payload, scale):
    """[m, n, k] CUDA E4M3 payload (unit column stride) with [m, n, words] packed int32 UE8M0 scales expanded per row,
    or [m, ceil(n / 128), ceil(k / 128)] FP32 block scales, read through their strides; returns contiguous BF16."""
    m, n, k = payload.shape
    out = torch.empty((m, n, k), dtype=torch.bfloat16, device=payload.device)
    if out.numel():
        tiles_k = triton.cdiv(k, _BLOCK_K)
        _dequant_kernel[(triton.cdiv(n, _BLOCK_N) * tiles_k, m)](
            payload,
            scale,
            out,
            n,
            k,
            tiles_k,
            payload.stride(0),
            payload.stride(1),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            PACKED=scale.dtype == torch.int32,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
        )
    return out
