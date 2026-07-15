import torch
import torch.nn.functional as F

from miles_plugins.models.inkling.options import inkling_opt


def _sconv_packed() -> bool:
    return inkling_opt("inkling_sconv_packed")


def _maybe_compile(fn):
    return torch.compile(fn, dynamic=True)


@_maybe_compile
def _swiglu_fwd(fc1_out, scale):
    g, u = torch.chunk(fc1_out.float(), 2, dim=-1)
    y = F.silu(g) * u
    if scale is not None:
        y = y * scale.float()
    return y.to(fc1_out.dtype)


@_maybe_compile
def _swiglu_bwd(fc1_out, scale, grad_out, need_dscale: bool):
    g, u = torch.chunk(fc1_out.float(), 2, dim=-1)
    s = torch.sigmoid(g)
    silu = g * s
    go = grad_out.float()
    if scale is not None:
        go = go * scale.float()
    d_g = go * u * (s + silu * (1 - s))
    d_u = go * silu
    d_scale = None
    if need_dscale:
        d_scale = (grad_out.float() * silu * u).sum(-1, keepdim=True).to(scale.dtype)
    return torch.cat([d_g, d_u], dim=-1).to(fc1_out.dtype), d_scale


class _SwigluFP32Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fc1_out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        ctx.save_for_backward(fc1_out, scale)
        return _swiglu_fwd(fc1_out, scale)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        fc1_out, scale = ctx.saved_tensors
        need_dscale = scale is not None and ctx.needs_input_grad[1]
        d_fc1, d_scale = _swiglu_bwd(fc1_out, scale, grad_out, need_dscale)
        return d_fc1, d_scale


def swiglu_fp32(fc1_out: torch.Tensor, per_token_scale: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 swiglu, one round back to bf16 (triton, bit-identical to serving silu_and_mul)."""
    from miles_plugins.models.inkling.ops.kernel.triton_swiglu import swiglu_fp32_triton

    return swiglu_fp32_triton(fc1_out, per_token_scale)


@_maybe_compile
def _sum2_fwd(a, b):
    return (a.float() + b.float()).to(a.dtype)


@_maybe_compile
def _sum3_fwd(a, b, c):
    return (a.float() + b.float() + c.float()).to(a.dtype)


class _SumFP32Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *ys: torch.Tensor) -> torch.Tensor:
        ctx.n = len(ys)
        if len(ys) == 2:
            return _sum2_fwd(*ys)
        if len(ys) == 3:
            return _sum3_fwd(*ys)
        out = ys[0].float()
        for y in ys[1:]:
            out = out + y.float()
        return out.to(ys[0].dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return (grad_out,) * ctx.n


def sum_fp32(ys: list[torch.Tensor]) -> torch.Tensor:
    """Σ ys in fp32, single round back (SGLang `_sum_dim0` parity)."""
    return _SumFP32Func.apply(*ys)


@_maybe_compile
def _sconv_fwd(x, weight):
    C = x.shape[1]
    xp = F.pad(x.float().t().unsqueeze(0), (weight.shape[-1] - 1, 0))
    y = x.float() + F.conv1d(xp, weight.float(), groups=C).squeeze(0).t()
    return y.to(x.dtype)


def sconv_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Residual depthwise causal conv on [T, C] in fp32, single round back (SGLang sconv parity)."""
    return _sconv_fwd(x, weight)


def sconv_fp32_packed(x: torch.Tensor, weight: torch.Tensor, seqlens) -> torch.Tensor:
    """Packed-sequence sconv: one full-length conv + exact boundary re-compute."""
    if x.is_cuda and inkling_opt("inkling_sconv_impl") == "triton":
        from miles_plugins.models.inkling.ops.kernel.triton_sconv import sconv_fp32_triton

        return sconv_fp32_triton(x, weight, seqlens)
    if seqlens is None or len(seqlens) <= 1:
        return sconv_fp32(x, weight)
    if not _sconv_packed():
        return torch.cat([sconv_fp32(s, weight) for s in x.split(list(seqlens))], 0)

    k = weight.shape[-1]
    T, C = x.shape
    xf = x.float()
    w = weight.float()
    xp = F.pad(xf.t().unsqueeze(0), (k - 1, 0))
    y = xf + F.conv1d(xp, w, groups=C).squeeze(0).t()  # fp32, interior rows correct

    if k > 1:
        dev = x.device
        sl = torch.as_tensor(list(seqlens), device=dev, dtype=torch.long)
        starts = sl.cumsum(0)[:-1]  # segment starts for segments 1..S-1
        lens = sl[1:]
        ds = torch.arange(k - 1, device=dev)  # boundary row offsets within a segment
        t_idx = starts.view(-1, 1) + ds.view(1, -1)  # [S-1, k-1] global row ids
        valid = ds.view(1, -1) < lens.view(-1, 1)
        xw = xf[t_idx.clamp(max=T - 1)]  # [S-1, k-1, C] rows s+0 .. s+k-2
        # W2[d, i, c] = w[c, k-1-(d-i)] for i<=d else 0: y[s+d] = x[s+d] + Σ_i xw[:,i]·W2[d,i]
        wt = w.squeeze(1).t()  # [k, C]
        W2 = x.new_zeros(k - 1, k - 1, C, dtype=torch.float32)
        for d in range(k - 1):
            for i in range(d + 1):
                W2[d, i] = wt[k - 1 - (d - i)]
        corr = torch.einsum("sic,dic->sdc", xw, W2)  # [S-1, k-1, C] fp32
        y = y.index_put((t_idx[valid],), (xw + corr)[valid])

    return y.to(x.dtype)
