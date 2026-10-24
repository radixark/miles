"""Grouped RMSNorm over n groups of C channels per token (Triton). grouped_rmsnorm_fwd launches the
forward; the backward kernel is launched inside the callers' autograd functions, fused with their grids."""

import torch
import triton
import triton.language as tl


def block_c(C: int) -> int:
    return triton.next_power_of_2(C)


@triton.jit(do_not_specialize=["T"])
def grouped_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    normed_ptr,  # fp32 out [T, n*C]
    rstd_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,  # streams
    C: tl.constexpr,  # per-stream hidden
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(normed_ptr + base + offs, x * rstd * (1.0 + w), mask=mask)
    tl.store(rstd_ptr + t * N + c, rstd)


@triton.jit(do_not_specialize=["T"])
def grouped_rmsnorm_bwd_kernel(
    x_ptr,
    w_ptr,
    rstd_ptr,
    dnormed_ptr,  # fp32 in [T, n*C]
    dx_ptr,  # out, x dtype
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dnormed_ptr + base + offs, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + t * N + c)
    g = dy * (1.0 + w)
    dot = tl.sum(g * x, axis=0)
    dx = rstd * g - x * (rstd * rstd * rstd) * (dot / C)
    tl.store(dx_ptr + base + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def grouped_rmsnorm_fwd(x2d: torch.Tensor, weight: torch.Tensor, n: int, eps: float):
    T, W = x2d.shape
    C = W // n
    normed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(T, n, dtype=torch.float32, device=x2d.device)
    if T > 0:
        grouped_rmsnorm_fwd_kernel[(T * n,)](x2d, weight, normed, rstd, T, N=n, C=C, EPS=eps, BLOCK_C=block_c(C))
    return normed, rstd
