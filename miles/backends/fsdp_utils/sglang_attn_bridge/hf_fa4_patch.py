"""Trainer attention through SGLang's vendored FlashAttention-4, called the way SGLang's ``fa4``
backend calls it (paged KV with page size 1, one split, batch-invariant), so the forward is bitwise
SGLang's decode. Gradients come from the deterministic ``flash_attn.cute`` backward."""

import torch

_batch_invariant_set = False


def _ensure_batch_invariant():
    global _batch_invariant_set
    if not _batch_invariant_set:
        from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import set_batch_invariant

        set_batch_invariant(True)
        _batch_invariant_set = True


class FA4AttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, sm_scale):
        from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache

        _ensure_batch_invariant()
        total, hkv, d = k.shape
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        page_table = cu_seqlens[:-1, None] + torch.arange(max_seqlen, device=q.device, dtype=torch.int32)[None, :]
        page_table = torch.minimum(page_table, cu_seqlens[1:, None] - 1).to(torch.int32)
        out, lse = flash_attn_with_kvcache(
            q=q,
            k_cache=k.view(total, 1, hkv, d),
            v_cache=v.view(total, 1, hkv, v.shape[-1]),
            page_table=page_table,
            cache_seqlens=lens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            return_softmax_lse=True,
            num_splits=1,
        )[:2]
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        from flash_attn.cute.interface import _flash_attn_bwd

        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            softmax_scale=ctx.sm_scale,
            causal=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=ctx.max_seqlen,
            max_seqlen_k=ctx.max_seqlen,
            deterministic=True,
        )[:3]
        return dq, dk, dv, None, None, None


def _fa4_attention(module, query, key, value, attention_mask, scaling: float, dropout: float = 0.0, **kwargs):
    """HF attention interface: query [B, H, S, D], key/value [B, Hkv, S, D] -> ([B, S, H, D], None)."""
    del attention_mask, dropout
    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S
    q = query.transpose(1, 2).reshape(total, num_heads, D).to(torch.bfloat16).contiguous()
    k = key.transpose(1, 2).reshape(total, num_kv_heads, D).to(torch.bfloat16).contiguous()
    v = value.transpose(1, 2).reshape(total, num_kv_heads, D).to(torch.bfloat16).contiguous()

    cu_seqlens = kwargs.get("cu_seq_lens_q")
    if cu_seqlens is None or B != 1:
        raise ValueError("the fa4 bridge needs one packed row with its document boundaries (cu_seq_lens_q)")
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)
    max_seqlen = kwargs.get("max_length_q")
    if max_seqlen is None:
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

    out = FA4AttnFunction.apply(q, k, v, cu_seqlens, int(max_seqlen), scaling)
    return out.view(B, S, num_heads, D), None


def apply_fa4_attention_patch(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["fa4"] = _fa4_attention
    model.config._attn_implementation = "fa4"
    return sum(1 for _, m in model.named_modules() if hasattr(m, "q_proj") and hasattr(m, "o_proj"))
