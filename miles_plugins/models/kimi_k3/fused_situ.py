"""Fused SiTU-and-mul for the Kimi-K3 expert activation (``--kimi-k3-situ-backend fused``).

The eager ``situ_and_mul`` runs about ten fp32 elementwise kernels forward and about fifteen in autograd
backward over the [tokens x top-k, 2 x 3072] expert intermediate, each a full memory pass. Here forward is one
pass (read x, write out) and backward one pass (read x and grad_out, write grad_x). Math is fp32 with the eager
function's operations; precise libdevice tanh/exp and IEEE division keep the forward within one bf16 ulp of eager
(bit-identical on almost every element). Backward uses the closed-form derivative, so its fp32 rounding differs
from autograd's (a few ulp of the bf16 result on a small fraction of elements).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _situ_terms(g, lin_in, BETA: tl.constexpr, LBETA: tl.constexpr):
    tg = libdevice.tanh(libdevice.div_rn(g, BETA))
    sg = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g))
    gate = BETA * tg * sg
    tl_ = libdevice.tanh(libdevice.div_rn(lin_in, LBETA))
    lin = LBETA * tl_
    return tg, sg, gate, tl_, lin


@triton.jit
def _situ_fwd_kernel(
    x_ptr, out_ptr, N, stride_x, stride_o, BETA: tl.constexpr, LBETA: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < N
    g = tl.load(x_ptr + row * stride_x + col, mask=m, other=0.0).to(tl.float32)
    lin_in = tl.load(x_ptr + row * stride_x + N + col, mask=m, other=0.0).to(tl.float32)
    _, _, gate, _, lin = _situ_terms(g, lin_in, BETA, LBETA)
    tl.store(out_ptr + row * stride_o + col, (gate * lin).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _situ_bwd_kernel(
    x_ptr,
    go_ptr,
    gx_ptr,
    N,
    stride_x,
    stride_go,
    stride_gx,
    BETA: tl.constexpr,
    LBETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < N
    g = tl.load(x_ptr + row * stride_x + col, mask=m, other=0.0).to(tl.float32)
    lin_in = tl.load(x_ptr + row * stride_x + N + col, mask=m, other=0.0).to(tl.float32)
    go = tl.load(go_ptr + row * stride_go + col, mask=m, other=0.0).to(tl.float32)
    tg, sg, gate, tl_, lin = _situ_terms(g, lin_in, BETA, LBETA)
    dgate = (1.0 - tg * tg) * sg + BETA * tg * sg * (1.0 - sg)
    dlin = 1.0 - tl_ * tl_
    tl.store(gx_ptr + row * stride_gx + col, (go * lin * dgate).to(gx_ptr.dtype.element_ty), mask=m)
    tl.store(gx_ptr + row * stride_gx + N + col, (go * gate * dlin).to(gx_ptr.dtype.element_ty), mask=m)


_BLOCK = 1024


class _FusedSitu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, linear_beta):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows, two_n = x2.shape
        n = two_n // 2
        out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
        if rows:
            _situ_fwd_kernel[(rows, triton.cdiv(n, _BLOCK))](
                x2, out, n, x2.stride(0), out.stride(0), BETA=float(beta), LBETA=float(linear_beta), BLOCK=_BLOCK
            )
        ctx.save_for_backward(x2)
        ctx.beta, ctx.linear_beta, ctx.shape = float(beta), float(linear_beta), shape
        return out.reshape(*shape[:-1], n)

    @staticmethod
    def backward(ctx, grad_out):
        (x2,) = ctx.saved_tensors
        rows, two_n = x2.shape
        n = two_n // 2
        go = grad_out.reshape(rows, n)
        if go.stride(-1) != 1:
            go = go.contiguous()
        gx = torch.empty_like(x2)
        if rows:
            _situ_bwd_kernel[(rows, triton.cdiv(n, _BLOCK))](
                x2,
                go,
                gx,
                n,
                x2.stride(0),
                go.stride(0),
                gx.stride(0),
                BETA=ctx.beta,
                LBETA=ctx.linear_beta,
                BLOCK=_BLOCK,
            )
        return gx.reshape(ctx.shape), None, None


def fused_situ_and_mul(x: torch.Tensor, beta: float = 4.0, linear_beta: float = 25.0) -> torch.Tensor:
    return _FusedSitu.apply(x, beta, linear_beta)
