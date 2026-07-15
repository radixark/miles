import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover
    flex_attention = create_block_mask = None


_FLEX_COMPILED = None


def flex_compiled():
    # max-autotune-no-cudagraphs required on B200/sm_100 (default bwd config exceeds 232KB smem)
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        assert flex_attention is not None, "flex_attention needs torch>=2.5 + inductor"
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")
    return _FLEX_COMPILED


def te_attention(q, k, v, rel_logits, seqlens, window_left, is_local, scale):
    # differentiable TE-DPA fwd; q [T,nh,hd], k/v [T,nkv,hd], rel_logits [T,nh,RE] -> [T,nh,hd]
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
    rb = rel_logits.permute(1, 0, 2)  # [nh, T, RE]
    qi = torch.arange(T, device=q.device).view(T, 1)
    ki = torch.arange(T, device=q.device).view(1, T)
    rd = qi - ki
    valid = (rd >= 0) & (rd < RE)
    idx = rd.clamp(0, RE - 1)
    bias = (torch.gather(rb, 2, idx.unsqueeze(0).expand(nh, T, T)) * valid.unsqueeze(0)).unsqueeze(0)  # [1,nh,T,T]
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

    from .rel_score_mod import get_inkling_relative_attention_score_mod

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
    return out  # [T, nh, hd]


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
            y = te_attention(qd, kd, vd, rd, sl, window_left, is_local, scale)
            gq, gk, gv, gr = torch.autograd.grad(y, [qd, kd, vd, rd], grad_out)
        return gq, gk, gv, gr, None, None, None, None, None


def inkling_fa4_attention(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent):
    """q [T,nh,hd] / k,v [T,nkv,hd] / rel_logits [T,nh,RE] fp32 -> [T,nh,hd]; fwd = sglang FA4, bwd recompute."""
    return _InklingFA4Attention.apply(q, k, v, rel_logits, seqlens, window_left, is_local, scale, rel_extent)
