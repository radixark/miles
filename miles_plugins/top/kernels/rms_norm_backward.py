"""Stands in for Transformer Engine's RMSNorm autograd: the forward is serving's RMSNorm or fused add-RMSNorm, which
has no backward and keeps no inverse-RMS statistics, so they are recomputed here (from the unrounded residual sum,
if any).
"""

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd


@triton.jit
def _prepare_residual(
    X,
    R,
    G,
    Z,
    DY,
    INV,
    H: tl.constexpr,
    XS: tl.constexpr,
    RS: tl.constexpr,
    GS: tl.constexpr,
    XC: tl.constexpr,
    RC: tl.constexpr,
    GC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H
    x = tl.load(X + row * XS + col * XC, mask, 0).to(tl.float32)
    r = tl.load(R + row * RS + col * RC, mask, 0).to(tl.float32)
    g = tl.load(G + row * GS + col * GC, mask, 0).to(tl.float32)
    z = x + r
    tl.store(Z + row * H + col, z, mask)
    tl.store(DY + row * H + col, g, mask)
    tl.store(INV + row, tl.rsqrt(tl.sum(z * z, 0) / H + EPS))


# N, the element count, is a runtime argument that only bounds the mask: a new token count reuses the kernel.
@triton.jit(do_not_specialize=["N"])
def _finish_residual(
    DZ,
    G,
    DX,
    DR,
    N,
    H: tl.constexpr,
    GS: tl.constexpr,
    GC: tl.constexpr,
    HAS_G: tl.constexpr,
    SEPARATE_DR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dz = tl.load(DZ + i, mask, 0)
    if HAS_G:
        dz += tl.load(G + (i // H) * GS + (i % H) * GC, mask, 0).to(tl.float32)
    tl.store(DX + i, dz, mask)
    if SEPARATE_DR:
        tl.store(DR + i, dz, mask)


def rms_norm_backward(x, weight, eps, grad_output, *, residual=None, grad_residual=None):
    """Use TE's derivative, preserving the unrounded residual sum."""
    if x.numel() == 0 or grad_output is None:
        dx = torch.zeros_like(x) if grad_residual is None else grad_residual.to(x.dtype)
        dr = None
        if residual is not None:
            dr = torch.zeros_like(residual) if grad_residual is None else grad_residual.to(residual.dtype)
        return dx, dr, torch.zeros_like(weight)

    if residual is None:
        dtype = torch.promote_types(x.dtype, weight.dtype)
        z = x.to(dtype).contiguous()
        w = weight.to(dtype).contiguous()
        dy = grad_output.to(dtype).contiguous()
        # SGLang does not expose inverse-RMS statistics. The extra TE output
        # is discarded and never replaces the serving forward result.
        _, _, inv_rms = rmsnorm_fwd(z, w, eps, None, None, TE_DType[dtype], 0, False)
        dx, dw = rmsnorm_bwd(dy, z, inv_rms, w, 0, False)
        return dx.to(x.dtype), None, dw.to(weight.dtype)

    # Accept the strided branch/residual views without contiguous copies.
    rows, h = x.shape
    z = torch.empty((rows, h), device=x.device, dtype=torch.float32)
    dy = torch.empty_like(z)
    inv_rms = torch.empty(rows, device=x.device, dtype=torch.float32)
    _prepare_residual[(rows,)](
        x,
        residual,
        grad_output,
        z,
        dy,
        inv_rms,
        h,
        x.stride(0),
        residual.stride(0),
        grad_output.stride(0),
        x.stride(1),
        residual.stride(1),
        grad_output.stride(1),
        eps,
        triton.next_power_of_2(h),
    )
    # TE requires matching activation, weight and output-gradient dtypes.
    dz, dw = rmsnorm_bwd(dy, z, inv_rms, weight.float().contiguous(), 0, False)
    dx = torch.empty((rows, h), device=x.device, dtype=x.dtype)
    dr = dx if residual.dtype == x.dtype else torch.empty_like(dx, dtype=residual.dtype)
    _finish_residual[(triton.cdiv(x.numel(), 1024),)](
        dz,
        grad_residual,
        dx,
        dr,
        x.numel(),
        h,
        grad_residual.stride(0) if grad_residual is not None else 0,
        grad_residual.stride(1) if grad_residual is not None else 0,
        grad_residual is not None,
        residual.dtype != x.dtype,
        1024,
    )
    return dx, dr, dw.to(weight.dtype)
