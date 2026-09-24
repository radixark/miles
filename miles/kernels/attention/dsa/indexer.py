"""DSA lightning indexer: per-query scores over a KV range, top-k selection, and the gathered
top-k scores with a fused backward.

Every function takes the packed layout: q [T, H, D] bf16, k [T_kv, D] bf16, weights [T, H] fp32,
and per-query KV ranges cu_seqlen_ks / cu_seqlen_ke [T] int32 (query t scores keys in [ks, ke)).
A batched sbhd caller runs one packed problem per batch element with `indexer_logits_sbhd`.
"""

import torch

from miles.kernels.attention.dsa.tilelang.indexer_bwd import indexer_bwd_interface
from miles.kernels.attention.dsa.tilelang.indexer_fwd import indexer_fwd_interface

_BWD_MIN_TOPK = 32


def causal_ranges(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed causal ranges: query t attends keys [segment_start(t), t + 1)."""
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends


def causal_ranges_compressed(seq_len_q: int, compress_ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal ranges over compressed KV: query p attends compressed groups [0, (p + 1) // ratio)."""
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return ks, ke


def indexer_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke) -> torch.Tensor:
    """Scores [T, T_kv] fp32; keys outside a query's range are -inf."""
    return indexer_fwd_interface(q, k, weights.float(), cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)


def indexer_logits_sbhd(q, k, weights, cu_seqlen_ks, cu_seqlen_ke) -> torch.Tensor:
    """q [S, B, H, D], k [S_kv, B, D], weights [S, B, H]; ranges are shared across the batch.
    Returns [B, S, S_kv]."""
    seqlen, batch, _, _ = q.shape
    logits = torch.empty([batch, seqlen, k.shape[0]], device=q.device, dtype=torch.float32)
    for b in range(batch):
        logits[b] = indexer_logits(
            q[:, b].contiguous(), k[:, b].contiguous(), weights[:, b].contiguous(), cu_seqlen_ks, cu_seqlen_ke
        )
    return logits


def gather_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    return torch.where(valid_mask, scores, float("-inf"))


def _pad_topk_pow2(topk_indices, grad_scores):
    topk = topk_indices.shape[-1]
    padded = max(_BWD_MIN_TOPK, 1 << (topk - 1).bit_length())
    if padded == topk:
        return topk_indices.contiguous(), grad_scores.contiguous()
    pad = padded - topk
    topk_indices = torch.nn.functional.pad(topk_indices, (0, pad), value=-1)
    grad_scores = torch.nn.functional.pad(grad_scores, (0, pad), value=0.0)
    return topk_indices.contiguous(), grad_scores.contiguous()


class _IndexerTopkScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, weights, logits, topk_indices):
        ctx.save_for_backward(q, k, weights, topk_indices)
        return gather_topk_scores(logits, topk_indices)

    @staticmethod
    def backward(ctx, grad_scores):
        q, k, weights, topk_indices = ctx.saved_tensors
        topk_indices, grad_scores = _pad_topk_pow2(topk_indices, grad_scores)
        grad_q, grad_w, grad_k = indexer_bwd_interface(q, weights, k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None


def indexer_topk_scores(q, k, weights, logits, topk_indices) -> torch.Tensor:
    """Scores [T, topk] gathered from `logits` at `topk_indices` (-1 = padding, scored -inf).
    The backward recomputes the selected scores from q / k / weights in one TileLang kernel."""
    return _IndexerTopkScores.apply(q, k, weights.float(), logits, topk_indices)


def lighting_indexer(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, topk: int, topk_fn):
    """Scores, top-k by `topk_fn(logits, topk)`, gathered scores. The caller owns the selection policy
    (e.g. routing replay), so the kernel holds no framework state. Returns (scores [T, topk],
    topk_indices [T, topk] int32)."""
    logits = indexer_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)
    topk_indices = topk_fn(logits, topk)
    return indexer_topk_scores(q, k, weights, logits, topk_indices), topk_indices
