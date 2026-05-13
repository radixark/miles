import torch

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        _, head_num, _ = index_q.shape
        logits = indexer_fwd_interface(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)
        if topk_indices is None:
            # Clamp k to logits' last-dim size: when the available context is
            # shorter than the configured index_topk (small CI tests, short
            # debug rollouts), torch.topk raises "selected index k out of range".
            # Then pad topk_indices back to the original `topk` width with -1
            # sentinels so the backward kernel's power-of-2 / multiple-of-32
            # assertions on shape[-1] still hold. pytorch_extract_topk_scores
            # and the TileLang bwd kernel both treat index==-1 as invalid.
            actual_topk = min(topk, logits.shape[-1])
            index_score, topk_indices = torch.topk(logits, actual_topk, dim=-1)
            topk_indices = topk_indices.to(torch.int32)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)
            if actual_topk < topk:
                padding_shape = (*topk_indices.shape[:-1], topk - actual_topk)
                topk_indices = torch.cat(
                    [topk_indices, topk_indices.new_full(padding_shape, -1)],
                    dim=-1,
                )

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    return IndexerFunction.apply(index_q, index_k, weights.squeeze(-1), cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices)


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
