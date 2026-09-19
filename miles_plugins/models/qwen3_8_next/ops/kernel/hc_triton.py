"""Triton hyper-connection kernels for Qwen3.8-Next (Qwen4Exp).

Numerical contract: bit-for-bit the same *policy* as the torch reference in
``ops/hc.py`` -- every reduction and elementwise step in fp32, one cast onto
the output dtype at the end. The torch path is the parity-verified reference;
this module exists because in real training the torch path is slow twice over:
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _grouped_rmsnorm_fwd_kernel(
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
    pid = tl.program_id(0)
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
def _grouped_rmsnorm_bwd_kernel(
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
    pid = tl.program_id(0)
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


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_fwd_kernel(
    gate_ptr,  # fp32 [T, n*C]
    normed_ptr,  # fp32 [T, n*C]
    out_ptr,  # out dtype [T, C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        acc += g * nrm
    acc = acc / N
    tl.store(out_ptr + t * C + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_bwd_kernel(
    dmixed_ptr,  # fp32 [T, C]
    gate_ptr,
    normed_ptr,
    dgate_ptr,  # fp32 out [T, n*C]
    dnormed_ptr,  # fp32 out [T, n*C] (mix contribution only)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    dm = tl.load(dmixed_ptr + t * C + offs, mask=mask, other=0.0) / N
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        tl.store(dgate_ptr + base + offs, dm * nrm, mask=mask)
        tl.store(dnormed_ptr + base + offs, dm * g, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_fwd_kernel(
    res_ptr,  # bf16 [T, n*C]
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    out_ptr,  # res dtype [T, n*C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(hpost_ptr + t * N + c)
    tl.store(out_ptr + base + offs, (r + a * y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_bwd_kernel(
    dout_ptr,  # bf16 [T, n*C] (incoming grad, cast on load)
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    dy_ptr,  # out y dtype [T, C]
    dhpost_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        do = tl.load(dout_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(hpost_ptr + t * N + c)
        dy += a * do
        tl.store(dhpost_ptr + t * N + c, tl.sum(do * y, axis=0))
    tl.store(dy_ptr + t * C + offs, dy.to(dy_ptr.dtype.element_ty), mask=mask)


def _block_c(C: int) -> int:
    return triton.next_power_of_2(C)


def _norm_fwd(x2d: torch.Tensor, weight: torch.Tensor, n: int, eps: float):
    T, W = x2d.shape
    C = W // n
    normed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(T, n, dtype=torch.float32, device=x2d.device)
    if T > 0:
        _grouped_rmsnorm_fwd_kernel[(T * n,)](x2d, weight, normed, rstd, T, N=n, C=C, EPS=eps, BLOCK_C=_block_c(C))
    return normed, rstd


def _gate_chain_fwd(normed, w_down, w_up, n):
    """fp32 gate = sigmoid(Wup silu(Wdown N / n)). Returns (gate, z1) fp32."""
    z1 = F.linear(normed, w_down.float()) / n
    gate = torch.sigmoid(F.linear(F.silu(z1), w_up.float()))
    return gate, z1


class _HCMixInject(torch.autograd.Function):
    """Fused mix + inject gate."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps):
        T, W = x2d.shape
        C = W // n
        normed, rstd = _norm_fwd(x2d, weight, n, eps)
        gate, _ = _gate_chain_fwd(normed, w_down, w_up, n)
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=x2d.device)
        if T > 0:
            _gate_mul_mean_fwd_kernel[(T,)](gate, normed, mixed, T, N=n, C=C, BLOCK_C=_block_c(C))
        if w_inject is not None:
            h_post = 2.0 * torch.sigmoid(F.linear(normed, w_inject.float()) / n)
        else:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps = n, eps
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        n, eps = ctx.n, ctx.eps
        T, W = x2d.shape
        C = W // n
        normed, _ = _norm_fwd(x2d, weight, n, eps)
        gate, z1 = _gate_chain_fwd(normed, w_down, w_up, n)
        s1 = F.silu(z1)

        dmixed = dmixed.float()
        dgate = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        dnormed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        if T > 0:
            _gate_mul_mean_bwd_kernel[(T,)](dmixed, gate, normed, dgate, dnormed, T, N=n, C=C, BLOCK_C=_block_c(C))

        dz2 = dgate * gate * (1.0 - gate)
        dw_up = dz2.t() @ s1
        ds1 = dz2 @ w_up.float()
        sig_z1 = torch.sigmoid(z1)
        dz1 = ds1 * sig_z1 * (1.0 + z1 * (1.0 - sig_z1))
        dz1 = dz1 / n  # z1 = (N @ w_down^T) / n
        dw_down = dz1.t() @ normed
        dnormed += dz1 @ w_down.float()

        dw_inject = None
        if w_inject is not None and dh_post is not None and dh_post.numel() > 0:
            u = F.linear(normed, w_inject.float()) / n
            sig_u = torch.sigmoid(u)
            du = dh_post.float() * 2.0 * sig_u * (1.0 - sig_u)
            du = du / n
            dw_inject = (du.t() @ normed).to(w_inject.dtype)
            dnormed += du @ w_inject.float()

        x_hat = (x2d.float().view(T, n, C) * rstd.unsqueeze(-1)).view(T, W)
        dweight = (dnormed * x_hat).sum(dim=0).to(weight.dtype)

        dx = torch.empty_like(x2d)
        if T > 0:
            _grouped_rmsnorm_bwd_kernel[(T * n,)](x2d, weight, rstd, dnormed, dx, T, N=n, C=C, BLOCK_C=_block_c(C))
        return (
            dx,
            dweight,
            dw_down.to(w_down.dtype),
            dw_up.to(w_up.dtype),
            dw_inject,
            None,
            None,
        )


class _HCCombine(torch.autograd.Function):
    """out = residual + h_post[:, c, None] * y  (flattened [T, n*C])."""

    @staticmethod
    def forward(ctx, residual2d, y2d, h_post, n):
        T, W = residual2d.shape
        C = W // n
        ctx.h_post_dtype = h_post.dtype
        h_post = h_post.float().contiguous()
        out = torch.empty_like(residual2d)
        if T > 0:
            _combine_fwd_kernel[(T * n,)](residual2d, y2d, h_post, out, T, N=n, C=C, BLOCK_C=_block_c(C))
        ctx.save_for_backward(y2d, h_post)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, dout):
        y2d, h_post = ctx.saved_tensors
        n = ctx.n
        T, W = dout.shape
        C = W // n
        dout = dout.contiguous()
        dy = torch.empty_like(y2d)
        dh_post = torch.empty(T, n, dtype=torch.float32, device=dout.device)
        if T > 0:
            _combine_bwd_kernel[(T,)](dout, y2d, h_post, dy, dh_post, T, N=n, C=C, BLOCK_C=_block_c(C))
        return dout, dy, dh_post.to(ctx.h_post_dtype), None


def hc_mix_inject_triton(x, weight, w_down, w_up, w_inject, n: int, eps: float):
    """Fused (grouped_gemma_rmsnorm -> hc_mix, hc_inject_gate)."""
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    mixed, h_post = _HCMixInject.apply(x2d, weight, w_down, w_up, w_inject, n, eps)
    mixed = mixed.reshape(*lead, -1)
    if w_inject is not None:
        h_post = h_post.reshape(*lead, n)
    return mixed, h_post


def hc_combine_triton(residual, block_output, h_post, n: int):
    """``X'_c = X_c + a_c * y`` -- fused elementwise. Shapes as ops.hc.hc_combine."""
    lead = residual.shape[:-1]
    r2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    y2d = block_output.reshape(-1, block_output.shape[-1]).contiguous()
    hp2d = h_post.reshape(-1, n)
    out = _HCCombine.apply(r2d, y2d, hp2d, n)
    return out.reshape(*lead, -1)
