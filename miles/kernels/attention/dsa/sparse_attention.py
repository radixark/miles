"""Sparse attention over top-k selected KV rows (DeepSeek Sparse Attention core).

q [B, S, H, d_v + d_tail] bf16, kv [B, S_kv, G, d_v + d_tail] bf16, indices [B, S, G, topk] int32 with
-1 as padding, attn_sink [H] fp32 or None. GLM-5 / DeepSeek-V3.2 pass a RoPE tail (d_tail > 0) and
no sink; DeepSeek-V4 passes a single latent (d_tail = 0, G = 1) and a learnable per-head sink.

The forward runs FlashMLA's sparse prefill kernel when it is installed and the shape is one it
supports (single latent group, d_v = 512; query heads are zero-padded up to 64 or 128), otherwise the
TileLang kernel. The backward is always the TileLang kernel; it takes the log-sum-exp in log2 space
with the sink folded in, which is what the TileLang forward emits and what FlashMLA's is converted to.
"""

import torch

from miles.kernels.attention.dsa.tilelang.sparse_attn_bwd import sparse_attn_bwd_interface
from miles.kernels.attention.dsa.tilelang.sparse_attn_fwd import sparse_attn_fwd_interface

try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

_LOG2_E = 1.4426950408889634
_TILELANG_TOPK_MULTIPLE = 64
_FLASH_MLA_TOPK_MULTIPLE = 128
_FLASH_MLA_HEADS = (64, 128)
_FLASH_MLA_HEAD_DIMS = (512, 576)
_FLASH_MLA_ARCH_MAJORS = (9, 10)


def _pad_topk_block(indices, multiple: int):
    topk = indices.shape[-1]
    padded = (topk + multiple - 1) // multiple * multiple
    if padded == topk:
        return indices.contiguous()
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1).contiguous()


def _default_forward_backend(q, kv, d_v: int) -> str:
    supported = (
        flash_mla_sparse_fwd is not None
        and torch.cuda.get_device_capability(q.device)[0] in _FLASH_MLA_ARCH_MAJORS
        and d_v == 512
        and q.shape[-1] in _FLASH_MLA_HEAD_DIMS
        and kv.shape[2] == 1
        and q.shape[2] <= _FLASH_MLA_HEADS[-1]
    )
    return "flash_mla" if supported else "tilelang"


def _flash_mla_forward(q, kv, indices, attn_sink, sm_scale):
    batch, seq_len, heads, _ = q.shape
    seq_len_kv = kv.shape[1]
    padded_heads = next(h for h in _FLASH_MLA_HEADS if h >= heads)
    q = q.reshape(batch * seq_len, heads, -1)
    sink = attn_sink
    if padded_heads != heads:
        q = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))
        sink = None if attn_sink is None else torch.nn.functional.pad(attn_sink, (0, padded_heads - heads))
    indices = _pad_topk_block(indices, _FLASH_MLA_TOPK_MULTIPLE)
    if batch == 1:
        flat_indices = indices.view(seq_len, 1, -1)
    else:
        offsets = torch.arange(batch, device=indices.device, dtype=indices.dtype).view(batch, 1, 1, 1) * seq_len_kv
        flat_indices = torch.where(indices >= 0, indices + offsets, -1).view(batch * seq_len, 1, -1)
    out, _, lse = flash_mla_sparse_fwd(
        q, kv.reshape(batch * seq_len_kv, 1, -1), flat_indices, sm_scale, d_v=512, attn_sink=sink
    )
    out, lse = out[:, :heads].contiguous(), lse[:, :heads] * _LOG2_E
    if attn_sink is not None:
        lse = torch.logaddexp2(lse, attn_sink.float().view(1, heads) * _LOG2_E)
    return out.reshape(batch, seq_len, heads, -1), lse.reshape(batch, seq_len, heads).contiguous()


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, attn_sink, sm_scale, d_v, backend):
        q, kv, indices = q.contiguous(), kv.contiguous(), _pad_topk_block(indices, _TILELANG_TOPK_MULTIPLE)
        if backend == "flash_mla":
            out, lse = _flash_mla_forward(q, kv, indices, attn_sink, sm_scale)
        else:
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
        return dq, dkv, None, d_sink, None, None, None


def sparse_attention(
    q, kv, indices, sm_scale: float, d_v: int | None = None, attn_sink=None, forward_backend: str | None = None
) -> torch.Tensor:
    """Returns out [B, S, H, d_v] bf16. ``d_v`` defaults to the full head dim (no RoPE tail).
    ``forward_backend`` (``"flash_mla"`` / ``"tilelang"``) overrides the default: FlashMLA on SM90 / SM100
    when installed and the shape fits, TileLang otherwise."""
    if d_v is None:
        d_v = q.shape[-1]
    backend = forward_backend or _default_forward_backend(q, kv, d_v)
    return _SparseAttention.apply(q, kv, indices, attn_sink, sm_scale, d_v, backend)
