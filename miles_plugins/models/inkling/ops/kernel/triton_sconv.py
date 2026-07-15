import torch
import triton
import triton.language as tl


@triton.jit
def _sconv_fwd_kernel(
    x_ptr,  # [T, D] input dtype
    w_ptr,  # [D, W]
    bos_ptr,  # [T] int32 — segment start per token
    y_ptr,  # [T, D]
    T,
    D,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    bos = tl.load(bos_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    x_cur = tl.load(x_ptr + t_off[:, None] * D + d_off[None, :], mask=td_mask, other=0)

    for iw in tl.static_range(W):
        shifted = t_off - (W - 1) + iw
        if iw == W - 1:
            tap = x_cur.to(tl.float32)
        else:
            in_x = shifted >= bos
            x_val = tl.load(
                x_ptr + shifted[:, None] * D + d_off[None, :],
                mask=in_x[:, None] & td_mask,
                other=0,
                eviction_policy="evict_last",
            )
            tap = x_val.to(tl.float32)
        w_val = tl.load(w_ptr + d_off * W + iw, mask=d_mask, other=0).to(tl.float32)
        acc += tap * w_val[None, :]

    acc += x_cur.to(tl.float32)  # residual LAST (sglang order)
    tl.store(y_ptr + t_off[:, None] * D + d_off[None, :], acc.to(y_ptr.dtype.element_ty), mask=td_mask)


@triton.jit
def _sconv_bwd_dx_kernel(
    go_ptr,  # [T, D] grad wrt y (input dtype)
    w_ptr,  # [D, W]
    eos_ptr,  # [T] int32 — segment end (exclusive) per token
    dx_ptr,  # [T, D]
    T,
    D,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # dx[t] = go[t] (residual) + Σ_iw w[:, iw] * go[t + (W-1) - iw]  (same segment)
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    eos = tl.load(eos_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for iw in tl.static_range(W):
        fwd_t = t_off + (W - 1) - iw
        in_seg = fwd_t < eos
        g_val = tl.load(
            go_ptr + fwd_t[:, None] * D + d_off[None, :],
            mask=in_seg[:, None] & td_mask,
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        w_val = tl.load(w_ptr + d_off * W + iw, mask=d_mask, other=0).to(tl.float32)
        acc += g_val * w_val[None, :]

    g_cur = tl.load(go_ptr + t_off[:, None] * D + d_off[None, :], mask=td_mask, other=0).to(tl.float32)
    acc += g_cur  # residual grad
    tl.store(dx_ptr + t_off[:, None] * D + d_off[None, :], acc.to(dx_ptr.dtype.element_ty), mask=td_mask)


def _bos_eos(seqlens, T, device):
    if seqlens is None or len(seqlens) <= 1:
        bos = torch.zeros(T, device=device, dtype=torch.int32)
        eos = torch.full((T,), T, device=device, dtype=torch.int32)
        return bos, eos
    sl = torch.as_tensor(list(seqlens), device=device, dtype=torch.int32)
    ends = sl.cumsum(0)
    starts = ends - sl
    bos = torch.repeat_interleave(starts, sl)
    eos = torch.repeat_interleave(ends, sl)
    return bos.to(torch.int32), eos.to(torch.int32)


class _SconvFP32Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, seqlens) -> torch.Tensor:
        # x [T, C] contiguous; weight [C, 1, W] (megatron _Sconv layout)
        T, D = x.shape
        W = weight.shape[-1]
        xc = x.contiguous()
        w2 = weight.reshape(D, W).contiguous()
        bos, eos = _bos_eos(seqlens, T, x.device)
        y = torch.empty_like(xc)
        if T > 0:
            grid = (triton.cdiv(T, 64), triton.cdiv(D, 128))
            _sconv_fwd_kernel[grid](xc, w2, bos, y, T, D, W=W, BLOCK_T=64, BLOCK_D=128)
        ctx.save_for_backward(xc, w2, bos, eos)
        ctx.w_shape = weight.shape
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xc, w2, bos, eos = ctx.saved_tensors
        T, D = xc.shape
        W = w2.shape[-1]
        go = grad_out.contiguous()
        dx = torch.empty_like(xc)
        if T > 0:
            grid = (triton.cdiv(T, 64), triton.cdiv(D, 128))
            _sconv_bwd_dx_kernel[grid](go, w2, eos, dx, T, D, W=W, BLOCK_T=64, BLOCK_D=128)
        # dw[d, iw] = Σ_t go32[t, d] * x32[t-(W-1)+iw, d] (in-segment) — torch reductions
        # (contiguous slices, no gather: x row for token t at shift s is just x[t-s])
        go32 = go.float()
        x32 = xc.float()
        dw = torch.empty(D, W, device=xc.device, dtype=torch.float32)
        ar = torch.arange(T, device=xc.device, dtype=torch.int32)
        for iw in range(W):
            shift = (W - 1) - iw
            if shift == 0:
                dw[:, iw] = (go32 * x32).sum(0)
            else:
                valid = (ar[shift:] - shift) >= bos[shift:]  # in-segment mask [T-shift]
                dw[:, iw] = (go32[shift:] * x32[: T - shift] * valid.unsqueeze(1)).sum(0)
        return dx, dw.reshape(ctx.w_shape).to(w2.dtype), None


def sconv_fp32_triton(x: torch.Tensor, weight: torch.Tensor, seqlens=None) -> torch.Tensor:
    """Packed-sequence residual sconv, forward bit-identical to sglang's kernel."""
    return _SconvFP32Triton.apply(x, weight, seqlens)
