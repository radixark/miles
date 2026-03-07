"""Minimal patch: replace HF attention kernel with SGLang Triton extend_attention_fwd_unified.

Instead of monkey-patching the entire attention forward (qkv proj, norm, rope, o_proj),
we register a custom attention backend via HF's ALL_ATTENTION_FUNCTIONS. This way HF
keeps its own qkv_proj, norm, rope, o_proj logic — we ONLY replace the core attention
computation (Q@K^T softmax V) with extend_attention_fwd_unified.

The extend_attention_fwd_unified kernel uses per-request indptrs, which naturally
gives batch-invariant results (each request is computed independently).
"""

import torch


def _sglang_triton_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in replacement for HF's attention_interface.

    Input:  query [B, num_heads, S, D], key/value [B, num_kv_heads, S, D]
    Output: attn_output [B, S, num_heads, D], None
    """
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    del attention_mask, scaling, dropout, kwargs

    B, num_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    total = B * S

    q = query.transpose(1, 2).contiguous().view(total, num_heads, D)
    k = key.transpose(1, 2).contiguous().view(total, num_kv_heads, D)
    v = value.transpose(1, 2).contiguous().view(total, num_kv_heads, D)

    # Force kernel inputs to bf16 right before extend_unified to match
    # the SGLang inference-side Triton backend behavior.
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    o = torch.empty_like(q)
    device = q.device
    qo_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * S
    kv_indptr = qo_indptr.clone()
    kv_indices = torch.arange(total, device=device, dtype=torch.int64)
    prefix_lens = torch.zeros(B, device=device, dtype=torch.int32)

    extend_attention_fwd_unified(
        q,
        o,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=S,
        is_causal=True,
    )

    attn_output = o.view(B, S, num_heads, D)
    return attn_output, None


def apply_sglang_triton_attention_patch(model):
    """Register SGLang Triton as attention backend and activate it on the model."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["triton"] = _sglang_triton_attention
    model.config._attn_implementation = "triton"

    patched = sum(
        1
        for _, m in model.named_modules()
        if hasattr(m, "q_proj") and hasattr(m, "o_proj")
    )
    return patched
