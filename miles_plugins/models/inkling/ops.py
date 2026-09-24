"""Inkling custom ops in one place: CP/SP helpers, precision-aligned fp32 ops (the triton kernels
live in miles.kernels.activation), and the attention backends (flex / TE reference / FA4 with
Inkling's relative position bias)."""

from collections.abc import Callable
from functools import cache

import megatron.core.parallel_state as ps
import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

from miles.kernels.activation.short_conv_fp32 import sconv_fp32_triton
from miles.kernels.activation.swiglu_fp32 import swiglu_fp32_triton


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover
    flex_attention = create_block_mask = None


def cp_world():
    """(cp_size, cp_rank, cp_group) — cp_size 1 means no context parallelism."""
    cp = ps.get_context_parallel_world_size()
    if cp <= 1:
        return 1, 0, None
    return cp, ps.get_context_parallel_rank(), ps.get_context_parallel_group()


def cp_all_gather(x, group, world):
    """all-gather [s, ...] -> [s*world, ...] in CP-rank order. Assumes CONTIGUOUS CP sharding
    (rank r holds tokens [r*s,(r+1)*s)) -> REQUIRES --allgather-cp (miles' default CP layout is
    zigzag/load-balanced; the provider asserts allgather_cp when cp>1)."""
    xs = [torch.empty_like(x) for _ in range(world)]
    torch.distributed.all_gather(xs, x.contiguous(), group=group)
    return torch.cat(xs, dim=0)


def inkling_sp_residual_conv(config, conv, x_sbh, seqlens):
    """Residual depthwise sconv on [s,b,h]. Under SP/CP the sequence is sharded, so a local-shard
    causal conv misses the left context -> gather -> conv on the full sequence -> take this rank's
    slice. Exact because the conv is residual. seqlens must be the FULL-sequence segment lengths."""
    sp = getattr(config, "sequence_parallel", False) and ps.get_tensor_model_parallel_world_size() > 1
    cp, cp_rank, cp_group = cp_world()
    x = gather_from_sequence_parallel_region(x_sbh, tensor_parallel_output_grad=False) if sp else x_sbh
    if cp > 1:
        x = cp_all_gather(x, cp_group, cp)
    s, b, h = x.shape
    x = conv(x.reshape(s * b, h), seqlens).reshape(s, b, h)
    if cp > 1:
        sloc = s // cp
        x = x[cp_rank * sloc : (cp_rank + 1) * sloc]
    return scatter_to_sequence_parallel_region(x) if sp else x


def seqlens_from_packed(packed_seq_params, T):
    """THD packing: per-segment token lengths (over the full post-SP-gather length T) from
    cu_seqlens_q, so attention/sconv/rel-bias never cross packed-sequence boundaries. Clip to T:
    keep whole segments, split the boundary one, trailing pad as its own segment."""
    if packed_seq_params is None or getattr(packed_seq_params, "cu_seqlens_q", None) is None:
        return None
    cu = packed_seq_params.cu_seqlens_q
    raw = [int(s) for s in (cu[1:] - cu[:-1]).tolist() if s > 0]
    seqlens, acc = [], 0
    for s in raw:
        if acc + s <= T:
            seqlens.append(s)
            acc += s
        else:
            if T - acc > 0:
                seqlens.append(T - acc)
            acc = T
            break
    if acc < T:
        seqlens.append(T - acc)
    return seqlens


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


def inkling_swiglu_fp32(fc1_out: torch.Tensor, per_token_scale: torch.Tensor | None = None) -> torch.Tensor:
    """fp32 swiglu, one round back to bf16 (triton, bit-identical to serving silu_and_mul)."""
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


def inkling_sum_fp32(ys: list[torch.Tensor]) -> torch.Tensor:
    """Σ ys in fp32, single round back (SGLang `_sum_dim0` parity)."""
    return _SumFP32Func.apply(*ys)


@_maybe_compile
def _sconv_fwd(x, weight):
    C = x.shape[1]
    xp = F.pad(x.float().t().unsqueeze(0), (weight.shape[-1] - 1, 0))
    y = x.float() + F.conv1d(xp, weight.float(), groups=C).squeeze(0).t()
    return y.to(x.dtype)


def inkling_sconv_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Residual depthwise causal conv on [T, C] in fp32, single round back (SGLang sconv parity)."""
    return _sconv_fwd(x, weight)


def inkling_sconv_fp32_packed(
    x: torch.Tensor, weight: torch.Tensor, seqlens, impl: str = "triton", packed: bool = False
) -> torch.Tensor:
    """Packed-sequence sconv. impl=triton is bit-identical to serving; packed runs the whole
    batch as one full-length conv + exact boundary re-compute instead of per-segment convs."""
    if x.is_cuda and impl == "triton":
        return sconv_fp32_triton(x, weight, seqlens)
    if seqlens is None or len(seqlens) <= 1:
        return inkling_sconv_fp32(x, weight)
    if not packed:
        return torch.cat([inkling_sconv_fp32(s, weight) for s in x.split(list(seqlens))], 0)

    k = weight.shape[-1]
    T, C = x.shape
    xf = x.float()
    w = weight.float()
    xp = F.pad(xf.t().unsqueeze(0), (k - 1, 0))
    y = xf + F.conv1d(xp, w, groups=C).squeeze(0).t()

    if k > 1:
        dev = x.device
        sl = torch.as_tensor(list(seqlens), device=dev, dtype=torch.long)
        starts = sl.cumsum(0)[:-1]
        lens = sl[1:]
        ds = torch.arange(k - 1, device=dev)
        t_idx = starts.view(-1, 1) + ds.view(1, -1)
        valid = ds.view(1, -1) < lens.view(-1, 1)
        xw = xf[t_idx.clamp(max=T - 1)]
        wt = w.squeeze(1).t()
        W2 = x.new_zeros(k - 1, k - 1, C, dtype=torch.float32)
        for d in range(k - 1):
            for i in range(d + 1):
                W2[d, i] = wt[k - 1 - (d - i)]
        corr = torch.einsum("sic,dic->sdc", xw, W2)
        y = y.index_put((t_idx[valid],), (xw + corr)[valid])

    return y.to(x.dtype)


try:
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK
except Exception as _import_error:
    cute = None
    Float32 = None
    SeqlenInfoQK = None
    _cute_import_error = _import_error
else:
    _cute_import_error = None


@cache
def get_inkling_relative_attention_score_mod(rel_extent: int) -> Callable:
    if cute is None or Float32 is None or SeqlenInfoQK is None:
        raise ImportError(
            "Inkling relative attention requires the vendored FA4 CUTE interface."
        ) from _cute_import_error

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


_FLEX_COMPILED = None


def flex_compiled():
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        assert flex_attention is not None, "flex_attention needs torch>=2.5 + inductor"
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")
    return _FLEX_COMPILED


def inkling_te_attention(q, k, v, rel_logits, seqlens, window_left, is_local, scale):
    import transformer_engine.pytorch as te

    T, nh, hd = q.shape
    nkv = k.shape[1]
    RE = rel_logits.shape[-1]
    dpa = te.DotProductAttention(
        num_attention_heads=nh,
        kv_channels=hd,
        num_gqa_groups=nkv,
        attention_dropout=0.0,
        qkv_format="sbhd",
        softmax_scale=scale,
    )
    rb = rel_logits.permute(1, 0, 2)
    qi = torch.arange(T, device=q.device).view(T, 1)
    ki = torch.arange(T, device=q.device).view(1, T)
    rd = qi - ki
    valid = (rd >= 0) & (rd < RE)
    idx = rd.clamp(0, RE - 1)
    bias = (torch.gather(rb, 2, idx.unsqueeze(0).expand(nh, T, T)) * valid.unsqueeze(0)).unsqueeze(0)
    win = (window_left, 0) if is_local else (-1, -1)
    if seqlens is not None and len(seqlens) > 1:
        seg = torch.repeat_interleave(
            torch.arange(len(seqlens), device=q.device), torch.tensor(seqlens, device=q.device)
        )
        cross = (seg.view(T, 1) != seg.view(1, T)).view(1, 1, T, T)
        bias = bias.masked_fill(cross, -1e9)
    ctx = dpa(
        q.unsqueeze(1).contiguous(),
        k.unsqueeze(1).contiguous(),
        v.unsqueeze(1).contiguous(),
        attn_mask_type="causal",
        window_size=win,
        core_attention_bias_type="post_scale_bias",
        core_attention_bias=bias.to(q.dtype),
    )
    return ctx.reshape(T, nh, hd)


def _fa4_fwd(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent):
    from flash_attn.cute.interface import flash_attn_varlen_func

    T = q.shape[0]
    sl = list(seqlens) if seqlens else [T]
    cu = torch.zeros(len(sl) + 1, device=q.device, dtype=torch.int32)
    cu[1:] = torch.tensor(sl, device=q.device, dtype=torch.int32).cumsum(0)
    win = (window_left, 0) if is_local else (-1, -1)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(sl),
        max_seqlen_k=max(sl),
        softmax_scale=scale,
        causal=True,
        window_size=win,
        score_mod=get_inkling_relative_attention_score_mod(rel_extent),
        aux_tensors=[rel_logits.contiguous().float()],
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


class _InklingFA4Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent):
        with torch.no_grad():
            out = _fa4_fwd(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent)
        ctx.save_for_backward(q, k, v, rel_logits)
        ctx.meta = (tuple(seqlens) if seqlens else None, window_left, is_local, scale, rel_extent)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, rel_logits = ctx.saved_tensors
        seqlens, window_left, is_local, scale, rel_extent = ctx.meta
        sl = list(seqlens) if seqlens else None
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            rd = rel_logits.detach().requires_grad_(True)
            y = inkling_te_attention(qd, kd, vd, rd, sl, window_left, is_local, scale)
            gq, gk, gv, gr = torch.autograd.grad(y, [qd, kd, vd, rd], grad_out)
        return gq, gk, gv, gr, None, None, None, None, None


def inkling_fa4_attention(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent):
    """q [T,nh,hd] / k,v [T,nkv,hd] / rel_logits [T,nh,RE] fp32 -> [T,nh,hd]; fwd = sglang FA4, bwd recompute."""
    return _InklingFA4Attention.apply(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent)
