import os

import torch
import triton
import triton.language as tl

FUSED = os.environ.get("MILES_DSV41_HC_FUSED", "0") == "1"


@triton.jit
def _mix_fwd(x_ptr, res_ptr, hres_ptr, hpost_ptr, out_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < d
    x = tl.load(x_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(N):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(N):
            r = tl.load(res_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            acc += tl.load(hres_ptr + (t * N + i) * N + j) * r
        out = tl.load(hpost_ptr + t * N + j) * x + acc
        tl.store(out_ptr + (t * N + j) * d + off, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mix_bwd(
    g_ptr,
    x_ptr,
    res_ptr,
    hres_ptr,
    hpost_ptr,
    gx_ptr,
    gres_ptr,
    ghres_ptr,
    ghpost_ptr,
    d,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    ghpost = tl.zeros([N], dtype=tl.float32)
    ghres = tl.zeros([N * N], dtype=tl.float32)
    for base in range(0, d, BLOCK):
        off = base + tl.arange(0, BLOCK)
        mask = off < d
        x = tl.load(x_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
        gx = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(N):
            g = tl.load(g_ptr + (t * N + j) * d + off, mask=mask, other=0.0).to(tl.float32)
            gx += tl.load(hpost_ptr + t * N + j) * g
            hits = tl.sum(tl.where(mask, g * x, 0.0))
            ghpost += tl.where(tl.arange(0, N) == j, hits, 0.0)
        tl.store(gx_ptr + t * d + off, gx.to(gx_ptr.dtype.element_ty), mask=mask)
        for i in tl.static_range(N):
            r = tl.load(res_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            gres = tl.zeros([BLOCK], dtype=tl.float32)
            for j in tl.static_range(N):
                g = tl.load(g_ptr + (t * N + j) * d + off, mask=mask, other=0.0).to(tl.float32)
                gres += tl.load(hres_ptr + (t * N + i) * N + j) * g
                hit = tl.sum(tl.where(mask, g * r, 0.0))
                ghres += tl.where(tl.arange(0, N * N) == i * N + j, hit, 0.0)
            tl.store(gres_ptr + (t * N + i) * d + off, gres.to(gres_ptr.dtype.element_ty), mask=mask)
    tl.store(ghpost_ptr + t * N + tl.arange(0, N), ghpost)
    tl.store(ghres_ptr + t * N * N + tl.arange(0, N * N), ghres)


class _HCMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, h_res, h_post, n):
        s, b, d = x.shape
        t = s * b
        ctx.dtypes = (x.dtype, residual.dtype, h_res.dtype, h_post.dtype)
        x = x.contiguous()
        residual = residual.contiguous()
        h_res = h_res.contiguous().float()
        h_post = h_post.contiguous().float()
        out = torch.empty(t, n, d, dtype=residual.dtype, device=x.device)
        block = 1024
        _mix_fwd[(t, triton.cdiv(d, block))](x, residual, h_res, h_post, out, d, N=n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(x, residual, h_res, h_post)
        ctx.n, ctx.shape = n, (s, b, d)
        return out.view(s, b, n * d)

    @staticmethod
    def backward(ctx, grad_out):
        x, residual, h_res, h_post = ctx.saved_tensors
        n = ctx.n
        s, b, d = ctx.shape
        t = s * b
        grad_out = grad_out.contiguous()
        gx = torch.empty_like(x)
        gres = torch.empty_like(residual)
        ghres = torch.empty(t, n, n, dtype=torch.float32, device=x.device)
        ghpost = torch.empty(t, n, dtype=torch.float32, device=x.device)
        _mix_bwd[(t,)](
            grad_out,
            x,
            residual,
            h_res,
            h_post,
            gx,
            gres,
            ghres,
            ghpost,
            d,
            N=n,
            BLOCK=1024,
            num_warps=4,
        )
        x_dt, res_dt, hres_dt, hpost_dt = ctx.dtypes
        return (
            gx.view(s, b, d).to(x_dt),
            gres.view(s, b, n * d).to(res_dt),
            ghres.view(s, b, n, n).to(hres_dt),
            ghpost.view(s, b, n).to(hpost_dt),
            None,
        )


def hc_mix(x, original_residual, h_res, h_post, n):
    """`h_post * x + einsum("sbij,sbid->sbjd", h_res, residual)`, flattened back to [s, b, n*d]."""
    return _HCMix.apply(x, original_residual, h_res, h_post, n)


def hc_mix_reference(x, original_residual, h_res, h_post, n):
    s, b, _ = original_residual.shape
    residual = original_residual.view(s, b, n, -1).float()
    mixed = torch.einsum("sbij,sbid->sbjd", h_res.float(), residual)
    out = h_post.float().unsqueeze(-1) * x.float().unsqueeze(2) + mixed
    return out.view(s, b, -1).to(original_residual.dtype)


@triton.jit
def _agg_fwd(x_ptr, pre_ptr, out_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < d
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(N):
        v = tl.load(x_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(pre_ptr + t * N + i) * v
    tl.store(out_ptr + t * d + off, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _agg_bwd(g_ptr, x_ptr, pre_ptr, gx_ptr, gpre_ptr, d, N: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    gpre = tl.zeros([N], dtype=tl.float32)
    for base in range(0, d, BLOCK):
        off = base + tl.arange(0, BLOCK)
        mask = off < d
        g = tl.load(g_ptr + t * d + off, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(N):
            v = tl.load(x_ptr + (t * N + i) * d + off, mask=mask, other=0.0).to(tl.float32)
            tl.store(
                gx_ptr + (t * N + i) * d + off,
                (tl.load(pre_ptr + t * N + i) * g).to(gx_ptr.dtype.element_ty),
                mask=mask,
            )
            hit = tl.sum(tl.where(mask, g * v, 0.0))
            gpre += tl.where(tl.arange(0, N) == i, hit, 0.0)
    tl.store(gpre_ptr + t * N + tl.arange(0, N), gpre)


class _Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre, n):
        s, b, nc = x.shape
        d = nc // n
        t = s * b
        ctx.dtypes = (x.dtype, pre.dtype)
        x = x.contiguous()
        pre = pre.contiguous().float()
        out = torch.empty(t, d, dtype=x.dtype, device=x.device)
        block = 1024
        _agg_fwd[(t, triton.cdiv(d, block))](x, pre, out, d, N=n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(x, pre)
        ctx.n, ctx.shape = n, (s, b, d)
        return out.view(s, b, d)

    @staticmethod
    def backward(ctx, grad_out):
        x, pre = ctx.saved_tensors
        n = ctx.n
        s, b, d = ctx.shape
        t = s * b
        grad_out = grad_out.contiguous()
        gx = torch.empty_like(x)
        gpre = torch.empty(t, n, dtype=torch.float32, device=x.device)
        _agg_bwd[(t,)](grad_out, x, pre, gx, gpre, d, N=n, BLOCK=1024, num_warps=4)
        x_dt, pre_dt = ctx.dtypes
        return gx.view(s, b, n * d).to(x_dt), gpre.view(s, b, n).to(pre_dt), None


def aggregate(x, pre, n):
    """`(pre.unsqueeze(-1) * x.view(s, b, n, d)).sum(dim=2)` in fp32, without the [s, b, n, d] temporary."""
    return _Aggregate.apply(x, pre, n)
