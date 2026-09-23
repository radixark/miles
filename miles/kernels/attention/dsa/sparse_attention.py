"""Sparse attention over top-k selected KV rows (DeepSeek Sparse Attention core).

q [B, S, H, d_v + d_tail] bf16, kv [B, S_kv, G, d_v + d_tail] bf16, indices [B, S, G, topk] int32 with
-1 as padding, attn_sink [H] fp32 or None. GLM-5 / DeepSeek-V3.2 pass a RoPE tail (d_tail > 0) and
no sink; DeepSeek-V4 passes a single latent (d_tail = 0, G = 1) and a learnable per-head sink.
"""

import torch

from miles.kernels.attention.dsa.tilelang.sparse_attn_bwd import sparse_attn_bwd_interface
from miles.kernels.attention.dsa.tilelang.sparse_attn_fwd import sparse_attn_fwd_interface

_LOG2_E = 1.44269504
_BLOCK_TOPK = 64


def _pad_topk(indices):
    topk = indices.shape[-1]
    padded = (topk + _BLOCK_TOPK - 1) // _BLOCK_TOPK * _BLOCK_TOPK
    if padded == topk:
        return indices.contiguous()
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1).contiguous()


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, attn_sink, sm_scale, d_v):
        q, kv, indices = q.contiguous(), kv.contiguous(), _pad_topk(indices)
        out, lse = sparse_attn_fwd_interface(q, kv, indices, attn_sink, d_v, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, indices, attn_sink, out, lse)
        ctx.sm_scale = sm_scale
        ctx.d_v = d_v
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, kv, indices, attn_sink, out, lse = ctx.saved_tensors
        dq, dkv, delta = sparse_attn_bwd_interface(
            q, kv, out, grad_out.contiguous(), indices, lse, ctx.d_v, sm_scale=ctx.sm_scale
        )
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp2(attn_sink.float() * _LOG2_E - lse)
            d_sink = -(delta * p_sink).sum(dim=(0, 1))
        return dq, dkv, None, d_sink, None, None


def sparse_attention(q, kv, indices, sm_scale: float, d_v: int | None = None, attn_sink=None) -> torch.Tensor:
    """Returns out [B, S, H, d_v] bf16. `d_v` defaults to the full head dim (no RoPE tail)."""
    if d_v is None:
        d_v = q.shape[-1]
    return _SparseAttention.apply(q, kv, indices, attn_sink, sm_scale, d_v)
