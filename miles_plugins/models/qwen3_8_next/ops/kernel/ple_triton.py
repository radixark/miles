"""Triton PLE kernels: fused gate chain and segment-aware causal depthwise conv.

Same numerical policy as the torch reference in ops/ple.py (which the sglang
parity runs verified): fp32 for every reduction and elementwise step, one cast
onto the output dtype. The kernels fuse

"""

import math

import torch
import triton
import triton.language as tl

from miles_plugins.models.qwen3_8_next.ops.kernel.hc_triton import _block_c, _grouped_rmsnorm_bwd_kernel, _norm_fwd


@triton.jit(do_not_specialize=["T"])
def _ple_gate_fwd_kernel(
    key_ptr,  # [T, n*C] (any float dtype)
    query_ptr,  # [T, n*C] hc_state
    value_ptr,  # [T, C]
    wk_ptr,
    wq_ptr,  # norm weights [n*C]
    gated_ptr,  # fp32 out [T, n*C]
    gate_ptr,  # fp32 out [T, n]
    rstdk_ptr,
    rstdq_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    SQRTC: tl.constexpr,
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

    k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)

    rk = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) / C + EPS)
    rq = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / C + EPS)
    kn = k * rk * (1.0 + wk)
    qn = q * rq * (1.0 + wq)

    score = tl.sum(kn * qn, axis=0) / SQRTC
    mag = tl.maximum(tl.abs(score), 1e-6)
    sgn = tl.where(score >= 0, 1.0, -1.0)
    gate = tl.sigmoid(sgn * tl.sqrt(mag))

    v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(gated_ptr + base + offs, gate * v, mask=mask)
    tl.store(gate_ptr + t * N + c, gate)
    tl.store(rstdk_ptr + t * N + c, rk)
    tl.store(rstdq_ptr + t * N + c, rq)


@triton.jit(do_not_specialize=["T"])
def _ple_gate_bwd_kernel(
    dgated_ptr,  # fp32 in [T, n*C]
    key_ptr,
    query_ptr,
    value_ptr,
    wk_ptr,
    wq_ptr,
    gate_ptr,
    rstdk_ptr,
    rstdq_ptr,
    dkey_ptr,  # out, key dtype [T, n*C]
    dquery_ptr,  # out, query dtype [T, n*C]
    dvalue_ptr,  # fp32 out [T, n, C] (summed over n by the host)
    dwk_partial_ptr,  # fp32 out [T, n*C] (dnormed_k * k_hat; host sums over T)
    dwq_partial_ptr,
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    SQRTC: tl.constexpr,
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
    sqrtC = SQRTC

    k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + t * N + c)
    rk = tl.load(rstdk_ptr + t * N + c)
    rq = tl.load(rstdq_ptr + t * N + c)
    dg_out = tl.load(dgated_ptr + base + offs, mask=mask, other=0.0)

    kn = k * rk * (1.0 + wk)
    qn = q * rq * (1.0 + wq)

    dgate = tl.sum(dg_out * v, axis=0)
    tl.store(dvalue_ptr + (t * N + c) * C + offs, dg_out * g, mask=mask)

    score = tl.sum(kn * qn, axis=0) / sqrtC
    mag = tl.maximum(tl.abs(score), 1e-6)
    du = dgate * g * (1.0 - g)
    ds = tl.where(tl.abs(score) > 1e-6, du / (2.0 * tl.sqrt(mag)), 0.0)

    dkn = qn * (ds / sqrtC)
    dqn = kn * (ds / sqrtC)

    tl.store(dwk_partial_ptr + base + offs, dkn * (k * rk), mask=mask)
    tl.store(dwq_partial_ptr + base + offs, dqn * (q * rq), mask=mask)

    gk = dkn * (1.0 + wk)
    dotk = tl.sum(gk * k, axis=0)
    dk = rk * gk - k * (rk * rk * rk) * (dotk / C)
    gq = dqn * (1.0 + wq)
    dotq = tl.sum(gq * q, axis=0)
    dq = rq * gq - q * (rq * rq * rq) * (dotq / C)

    tl.store(dkey_ptr + base + offs, dk.to(dkey_ptr.dtype.element_ty), mask=mask)
    tl.store(dquery_ptr + base + offs, dq.to(dquery_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T", "W"])
def _ple_conv_fwd_kernel(
    normed_ptr,  # fp32 [T, W]
    gated_ptr,  # fp32 [T, W]
    convw_ptr,  # [W, K] (weight[:, 0, :] contiguous)
    segstart_ptr,  # int32 [T]
    out_ptr,  # out dtype [T, W]
    conv_ptr,  # fp32 out [T, W]  pre-SiLU conv result (saved for bwd)
    T,
    W,
    K: tl.constexpr,
    DIL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    wb = tl.program_id(1)
    if t >= T:
        return
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < W
    seg_lo = tl.load(segstart_ptr + t)

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for j in tl.static_range(K):
        src = t - (K - 1 - j) * DIL
        wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        if src >= 0:
            ok = src >= seg_lo
            x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
            acc += wgt * x
    tl.store(conv_ptr + t * W + offs, acc, mask=mask)
    silu = acc * tl.sigmoid(acc)
    gt = tl.load(gated_ptr + t * W + offs, mask=mask, other=0.0)
    tl.store(out_ptr + t * W + offs, (gt + silu).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T", "W"])
def _ple_conv_bwd_kernel(
    dout_ptr,  # incoming grad [T, W] (any float dtype)
    conv_ptr,  # fp32 [T, W] pre-SiLU
    normed_ptr,  # fp32 [T, W]
    convw_ptr,  # [W, K]
    segstart_ptr,
    segend_ptr,  # int32 [T] (exclusive)
    dnormed_ptr,  # fp32 out [T, W]
    dconvw_ptr,  # fp32 out [W, K] via atomics
    dgated_add_ptr,  # fp32 out [T, W]  (dout passthrough for the residual)
    T,
    W,
    K: tl.constexpr,
    DIL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    wb = tl.program_id(1)
    if t >= T:
        return
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < W
    seg_lo = tl.load(segstart_ptr + t)
    seg_hi = tl.load(segend_ptr + t)

    do = tl.load(dout_ptr + t * W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dgated_add_ptr + t * W + offs, do, mask=mask)

    cv = tl.load(conv_ptr + t * W + offs, mask=mask, other=0.0)
    sig = tl.sigmoid(cv)
    dconv = do * sig * (1.0 + cv * (1.0 - sig))

    for j in tl.static_range(K):
        src = t - (K - 1 - j) * DIL
        if src >= 0:
            ok = src >= seg_lo
            x = tl.load(normed_ptr + src * W + offs, mask=mask & ok, other=0.0)
            tl.atomic_add(dconvw_ptr + offs * K + j, dconv * x, mask=mask & ok)

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for j in tl.static_range(K):
        dst = t + (K - 1 - j) * DIL
        wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        if dst < T:
            ok = dst < seg_hi
            do2 = tl.load(dout_ptr + dst * W + offs, mask=mask & ok, other=0.0).to(tl.float32)
            cv2 = tl.load(conv_ptr + dst * W + offs, mask=mask & ok, other=0.0)
            sig2 = tl.sigmoid(cv2)
            acc += wgt * do2 * sig2 * (1.0 + cv2 * (1.0 - sig2))
    tl.store(dnormed_ptr + t * W + offs, acc, mask=mask)


def _seg_bounds(T: int, cu_seqlens, device):
    if cu_seqlens is None:
        lo = torch.zeros(T, dtype=torch.int32, device=device)
        hi = torch.full((T,), T, dtype=torch.int32, device=device)
        return lo, hi
    cu = cu_seqlens.to(torch.long)
    lens = (cu[1:] - cu[:-1]).clamp_min(0)
    lo = torch.repeat_interleave(cu[:-1], lens).to(torch.int32)
    hi = torch.repeat_interleave(cu[1:], lens).to(torch.int32)
    return lo, hi


class _PLEGateConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hc_state, key, value, wk, wq, wc, conv_w, n, eps, dilation, cu_seqlens):
        T, W = hc_state.shape
        C = W // n
        Kk = conv_w.shape[-1]
        dev = hc_state.device

        gated = torch.empty(T, W, dtype=torch.float32, device=dev)
        gate = torch.empty(T, n, dtype=torch.float32, device=dev)
        rstdk = torch.empty(T, n, dtype=torch.float32, device=dev)
        rstdq = torch.empty(T, n, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_fwd_kernel[(T * n,)](
                key,
                hc_state,
                value,
                wk,
                wq,
                gated,
                gate,
                rstdk,
                rstdq,
                T,
                N=n,
                C=C,
                EPS=eps,
                SQRTC=math.sqrt(C),
                BLOCK_C=_block_c(C),
            )

        normed, rstdc = _norm_fwd(gated, wc, n, eps)

        seg_lo, seg_hi = _seg_bounds(T, cu_seqlens, dev)
        convw2d = conv_w.reshape(W, Kk).contiguous()
        out = torch.empty(T, W, dtype=hc_state.dtype, device=dev)
        conv_pre = torch.empty(T, W, dtype=torch.float32, device=dev)
        BW = 256
        if T > 0:
            _ple_conv_fwd_kernel[(T, triton.cdiv(W, BW))](
                normed,
                gated,
                convw2d,
                seg_lo,
                out,
                conv_pre,
                T,
                W,
                K=Kk,
                DIL=dilation,
                BLOCK_W=BW,
            )

        ctx.save_for_backward(
            hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre
        )
        ctx.dims = (n, eps, dilation, Kk, conv_w.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        (hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre) = (
            ctx.saved_tensors
        )
        n, eps, dilation, Kk, conv_w_dtype = ctx.dims
        T, W = hc_state.shape
        C = W // n
        dev = hc_state.device
        dout = dout.contiguous()

        gated = torch.empty(T, W, dtype=torch.float32, device=dev)
        _g = torch.empty(T, n, dtype=torch.float32, device=dev)
        _rk = torch.empty(T, n, dtype=torch.float32, device=dev)
        _rq = torch.empty(T, n, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_fwd_kernel[(T * n,)](
                key,
                hc_state,
                value,
                wk,
                wq,
                gated,
                _g,
                _rk,
                _rq,
                T,
                N=n,
                C=C,
                EPS=eps,
                SQRTC=math.sqrt(C),
                BLOCK_C=_block_c(C),
            )
        normed, _ = _norm_fwd(gated, wc, n, eps)

        dnormed = torch.empty(T, W, dtype=torch.float32, device=dev)
        dconvw = torch.zeros(W, Kk, dtype=torch.float32, device=dev)
        dgated = torch.empty(T, W, dtype=torch.float32, device=dev)
        BW = 256
        if T > 0:
            _ple_conv_bwd_kernel[(T, triton.cdiv(W, BW))](
                dout,
                conv_pre,
                normed,
                convw2d,
                seg_lo,
                seg_hi,
                dnormed,
                dconvw,
                dgated,
                T,
                W,
                K=Kk,
                DIL=dilation,
                BLOCK_W=BW,
            )

        x_hat = (gated.view(T, n, C) * rstdc.unsqueeze(-1)).view(T, W)
        dwc = (dnormed * x_hat).sum(dim=0).to(wc.dtype)
        dgated_norm = torch.empty(T, W, dtype=torch.float32, device=dev)
        if T > 0:
            _grouped_rmsnorm_bwd_kernel[(T * n,)](
                gated, wc, rstdc, dnormed, dgated_norm, T, N=n, C=C, BLOCK_C=_block_c(C)
            )
        dgated += dgated_norm

        dkey = torch.empty_like(key)
        dquery = torch.empty_like(hc_state)
        dvalue_pern = torch.empty(T, n, C, dtype=torch.float32, device=dev)
        dwk_part = torch.empty(T, W, dtype=torch.float32, device=dev)
        dwq_part = torch.empty(T, W, dtype=torch.float32, device=dev)
        if T > 0:
            _ple_gate_bwd_kernel[(T * n,)](
                dgated,
                key,
                hc_state,
                value,
                wk,
                wq,
                gate,
                rstdk,
                rstdq,
                dkey,
                dquery,
                dvalue_pern,
                dwk_part,
                dwq_part,
                T,
                N=n,
                C=C,
                SQRTC=math.sqrt(C),
                BLOCK_C=_block_c(C),
            )
        dvalue = dvalue_pern.sum(dim=1).to(value.dtype)
        dwk = dwk_part.sum(dim=0).to(wk.dtype)
        dwq = dwq_part.sum(dim=0).to(wq.dtype)
        dconv_w = dconvw.view(W, 1, Kk).to(conv_w_dtype)

        return (dquery, dkey, dvalue, dwk, dwq, dwc, dconv_w, None, None, None, None)


def ple_gate_conv_triton(
    hc_state,
    key,
    value,
    norm_key_w,
    norm_query_w,
    norm_conv_w,
    conv1d_weight,
    n: int,
    eps: float,
    dilation: int,
    cu_seqlens,
):
    """Full PLE increment (gate chain + norm + causal conv + SiLU + residual)."""
    return _PLEGateConv.apply(
        hc_state.contiguous(),
        key.contiguous(),
        value.contiguous(),
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        conv1d_weight,
        n,
        eps,
        dilation,
        cu_seqlens,
    )
