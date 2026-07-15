import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    x_ptr,  # [M, 2N] block layout [g | u], input dtype
    scale_ptr,  # [M] fp32 or dummy
    y_ptr,  # [M, N] output dtype
    M,
    N,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    g = tl.load(x_ptr + offs_m[:, None] * (2 * N) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + offs_m[:, None] * (2 * N) + N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    if HAS_SCALE:
        sc = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        y = y * sc[:, None]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    x_ptr,  # [M, 2N] block layout, input dtype
    scale_ptr,  # [M] fp32 or dummy
    go_ptr,  # [M, N] grad_out, output dtype
    dx_ptr,  # [M, 2N] d_fc1 (input dtype)
    dscale_ptr,  # [M] fp32 partial (or dummy)
    M,
    N,
    HAS_SCALE: tl.constexpr,
    NEED_DSCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
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
        d_g = gos * u * (s + silu * (1 - s))
        d_u = gos * silu
        tl.store(dx_ptr + row * (2 * N) + offs_n, d_g, mask=mask_n)
        tl.store(dx_ptr + row * (2 * N) + N + offs_n, d_u, mask=mask_n)
        if NEED_DSCALE:
            acc += tl.where(mask_n, go * silu * u, 0.0)
    if NEED_DSCALE:
        tl.store(dscale_ptr + row, tl.sum(acc))


class _SwigluFP32Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fc1_out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        shp = fc1_out.shape
        x = fc1_out.reshape(-1, shp[-1]).contiguous()
        M, N2 = x.shape
        N = N2 // 2
        sc = scale.reshape(-1).contiguous().float() if scale is not None else x.new_empty(1, dtype=torch.float32)
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
        if M > 0:
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
            _swiglu_fwd_kernel[grid](x, sc, y, M, N, HAS_SCALE=scale is not None, BLOCK_M=32, BLOCK_N=128)
        ctx.save_for_backward(fc1_out, scale)
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
                HAS_SCALE=scale is not None,
                NEED_DSCALE=need_dscale,
                BLOCK_N=256,
            )
        d_scale = dsc.reshape(scale.shape).to(scale.dtype) if need_dscale else None
        return dx.reshape(shp), d_scale


def swiglu_fp32_triton(fc1_out: torch.Tensor, per_token_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Triton swiglu, forward bit-identical to sglang's silu_and_mul kernel."""
    return _SwigluFP32Triton.apply(fc1_out, per_token_scale)
