import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, f_ptr, n_rows, seqlen, inner, row_stride, HALF: tl.constexpr, INVERSE: tl.constexpr, ROWS: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    pos = (rows // inner) % seqlen
    k = tl.arange(0, HALF)
    x_offs = rows[:, None] * row_stride + 2 * k[None, :]
    mask = rmask[:, None] & (k[None, :] < HALF)
    xr = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + x_offs + 1, mask=mask, other=0.0).to(tl.float32)
    f_offs = pos[:, None] * (2 * HALF) + 2 * k[None, :]
    fr = tl.load(f_ptr + f_offs, mask=mask, other=0.0)
    fi = tl.load(f_ptr + f_offs + 1, mask=mask, other=0.0)
    if INVERSE:
        fi = -fi
    outr = xr * fr - xi * fi
    outi = xr * fi + xi * fr
    tl.store(x_ptr + x_offs, outr.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(x_ptr + x_offs + 1, outi.to(x_ptr.dtype.element_ty), mask=mask)


def row_stride_or_none(x: torch.Tensor):
    """Row stride of a contiguous tensor or a last-dim slice of one, else None."""
    if x.stride(-1) != 1:
        return None
    if x.ndim == 1:
        return x.shape[-1]
    row_stride = x.stride(-2)
    expected = row_stride
    for d in range(x.ndim - 3, -1, -1):
        expected *= x.shape[d + 1]
        if x.stride(d) != expected:
            return None
    return row_stride


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place RoPE on the last dim of x ([b, s, d] or [b, s, h, d]); `freqs_cis` is complex [s, d/2]."""
    dim = x.shape[-1]
    half = dim // 2
    assert freqs_cis.shape[-1] == half
    row_stride = row_stride_or_none(x)
    if row_stride is None:
        # rows of a permuted view have no single stride: rotate a contiguous copy, then write it back in place
        x.copy_(apply_rotary_emb(x.contiguous(), freqs_cis, inverse))
        return x
    seqlen = x.shape[1]
    inner = x.shape[2] if x.ndim == 4 else 1
    f = torch.view_as_real(freqs_cis.contiguous()).contiguous()
    n_rows = x.numel() // dim
    rows_per_prog = 32
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    _rope_kernel[grid](x, f, n_rows, seqlen, inner, row_stride, HALF=half, INVERSE=inverse, ROWS=rows_per_prog)
    return x
