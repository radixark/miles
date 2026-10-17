"""fp32 SwiGLU with an optional per-token scale and clamp; forward is bit-identical to sglang's silu_and_mul.

With ``limit``, ``min(silu(gate), limit) * clamp(up, -limit, limit)``: the clamp applies to SiLU's output, as in
sglang's SiluClampAndMul (vLLM's swiglustep), not to its input as Megatron's ``activation_func_clamp_value`` does.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    M,
    N,
    LIMIT,
    HAS_SCALE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # int64: row offsets reach M * 2N, past 2**31 for a 13,824-wide MLP above ~77K tokens
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    g = tl.load(x_ptr + offs_m[:, None] * (2 * N) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + offs_m[:, None] * (2 * N) + N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        y = tl.minimum(g * tl.sigmoid(g), LIMIT) * tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    else:
        y = g * tl.sigmoid(g) * u
    if HAS_SCALE:
        sc = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        y = y * sc[:, None]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    x_ptr,
    scale_ptr,
    go_ptr,
    dx_ptr,
    dscale_ptr,
    M,
    N,
    LIMIT,
    HAS_SCALE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    NEED_DSCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= M:
        return
    sc = 1.0
    if HAS_SCALE:
        sc = tl.load(scale_ptr + row).to(tl.float32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in tl.range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        g = tl.load(x_ptr + row * (2 * N) + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * (2 * N) + N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        go = tl.load(go_ptr + row * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        silu = g * s
        gos = go * sc
        if HAS_LIMIT:
            # torch.clamp passes the gradient where the input sits inside the bounds, edges included
            silu_c = tl.minimum(silu, LIMIT)
            u_c = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
            d_g = tl.where(silu <= LIMIT, gos * u_c * (s + silu * (1 - s)), 0.0)
            d_u = tl.where((u >= -LIMIT) & (u <= LIMIT), gos * silu_c, 0.0)
        else:
            silu_c = silu
            u_c = u
            d_g = gos * u * (s + silu * (1 - s))
            d_u = gos * silu
        tl.store(dx_ptr + row * (2 * N) + offs_n, d_g, mask=mask_n)
        tl.store(dx_ptr + row * (2 * N) + N + offs_n, d_u, mask=mask_n)
        if NEED_DSCALE:
            acc += tl.where(mask_n, go * silu_c * u_c, 0.0)
    if NEED_DSCALE:
        tl.store(dscale_ptr + row, tl.sum(acc))


class _SwigluFP32Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fc1_out: torch.Tensor, scale: torch.Tensor | None, limit: float | None) -> torch.Tensor:
        shp = fc1_out.shape
        x = fc1_out.reshape(-1, shp[-1]).contiguous()
        M, N2 = x.shape
        N = N2 // 2
        sc = scale.reshape(-1).contiguous().float() if scale is not None else x.new_empty(1, dtype=torch.float32)
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
        if M > 0:
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
            _swiglu_fwd_kernel[grid](
                x,
                sc,
                y,
                M,
                N,
                0.0 if limit is None else limit,
                HAS_SCALE=scale is not None,
                HAS_LIMIT=limit is not None,
                BLOCK_M=32,
                BLOCK_N=128,
            )
        ctx.save_for_backward(fc1_out, scale)
        ctx.limit = limit
        return y.reshape(*shp[:-1], N)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        fc1_out, scale = ctx.saved_tensors
        shp = fc1_out.shape
        x = fc1_out.reshape(-1, shp[-1]).contiguous()
        M, N2 = x.shape
        N = N2 // 2
        go = grad_out.reshape(-1, N).contiguous()
        sc = scale.reshape(-1).contiguous().float() if scale is not None else x.new_empty(1, dtype=torch.float32)
        need_dscale = scale is not None and ctx.needs_input_grad[1]
        dx = torch.empty_like(x)
        dsc = (
            torch.empty(M, device=x.device, dtype=torch.float32)
            if need_dscale
            else x.new_empty(1, dtype=torch.float32)
        )
        if M > 0:
            _swiglu_bwd_kernel[(M,)](
                x,
                sc,
                go,
                dx,
                dsc,
                M,
                N,
                0.0 if ctx.limit is None else ctx.limit,
                HAS_SCALE=scale is not None,
                HAS_LIMIT=ctx.limit is not None,
                NEED_DSCALE=need_dscale,
                BLOCK_N=256,
            )
        d_scale = dsc.reshape(scale.shape).to(scale.dtype) if need_dscale else None
        return dx.reshape(shp), d_scale, None


def swiglu_fp32(
    fc1_out: torch.Tensor, per_token_scale: torch.Tensor | None = None, limit: float | None = None
) -> torch.Tensor:
    """Triton swiglu, forward bit-identical to sglang's silu_and_mul (and, with ``limit``, SiluClampAndMul)."""
    return _SwigluFP32Triton.apply(fc1_out, per_token_scale, limit)
