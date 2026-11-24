"""FP32 -> BF16 stochastic rounding with Transformer Engine's Philox4x32-10 random bits.

Random bits follow TE's NVFP4 stochastic rounding: TE's ``curanddx`` state ``init(seed,
subsequence, offset)`` followed by ``generate4()`` is Philox4x32-10 with key = (seed lo, seed hi)
and counter = (offset lo, offset hi, subsequence lo, subsequence hi), which is ``tl.philox`` with
those four counter words. One ``generate4()`` covers 8 consecutive elements: element ``8 * s + j``
uses subsequence ``s`` and bits ``[16 * (j % 2), 16 * (j % 2) + 16)`` of output word ``j // 2``,
in the way TE's ``cvt.rs`` paths take consecutive bit fields of ``rbits``.

Rounding adds those 16 bits to the 16 low bits that BF16 drops and truncates, so a value rounds
away from zero with probability equal to its distance from the BF16 value below it, in units of
the BF16 spacing. Infinities stay infinite and NaN gets the encoding of the round-to-nearest cast.
"""

import torch
import triton
import triton.language as tl

ELEMENTS_PER_PHILOX_BLOCK = 8
_BLOCK = 4096


@triton.jit(do_not_specialize=["numel", "seed", "offset_lo", "offset_hi"])
def _stochastic_round_bf16_kernel(src_ptr, dst_ptr, numel, seed, offset_lo, offset_hi, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    subsequence = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    c2 = (subsequence & 0xFFFFFFFF).to(tl.uint32)
    c3 = (subsequence >> 32).to(tl.uint32)
    c0 = tl.zeros_like(c2) + offset_lo.to(tl.uint32)
    c1 = tl.zeros_like(c2) + offset_hi.to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, c2, c3)
    # Rows of [r0, r1, r2, r3], then the low and high half of each word: 8 x 16 bits per subsequence.
    words = tl.join(tl.join(r0, r2), tl.join(r1, r3))
    random16 = tl.reshape(tl.join(words & 0xFFFF, words >> 16), (BLOCK,))

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    # Carry from the 16 dropped bits into the kept ones with probability (dropped bits) / 2^16.
    rounded = ((x.to(tl.uint32, bitcast=True) + random16) >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    rounded = tl.where(x != x, x.to(tl.bfloat16), rounded)
    tl.store(dst_ptr + offsets, rounded, mask=mask)


def stochastic_round_to_bf16(src: torch.Tensor, out: torch.Tensor, *, seed: int, offset: int) -> torch.Tensor:
    """Write ``src`` (contiguous FP32) into ``out`` (contiguous BF16, same device and size), rounded stochastically.

    ``seed`` (< 2^64) is the Philox key and ``offset`` (< 2^64) the Philox offset; element ``i`` uses
    subsequence ``i // 8``. The same arguments always produce the same bits.
    """
    if src.dtype != torch.float32 or out.dtype != torch.bfloat16:
        raise TypeError(f"expected float32 -> bfloat16, got {src.dtype} -> {out.dtype}")
    if src.device != out.device or src.numel() != out.numel():
        raise ValueError(f"src {src.numel()} on {src.device} does not match out {out.numel()} on {out.device}")
    if not (src.is_contiguous() and out.is_contiguous()):
        raise ValueError("src and out must be contiguous")
    if not (0 <= seed < 1 << 64 and 0 <= offset < 1 << 64):
        raise ValueError(f"seed {seed} and offset {offset} must be unsigned 64-bit values")
    numel = src.numel()
    if numel:
        grid = (triton.cdiv(numel, _BLOCK),)
        _stochastic_round_bf16_kernel[grid](
            src, out, numel, seed, offset & 0xFFFFFFFF, offset >> 32, BLOCK=_BLOCK, num_warps=8
        )
    return out


__all__ = ["ELEMENTS_PER_PHILOX_BLOCK", "stochastic_round_to_bf16"]
