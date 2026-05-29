import torch

from miles.utils.replay_base import indexer_replay_manager

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def _original_topk(logits, topk):
    score, indices = torch.topk(logits, topk, dim=-1)
    indices = indices.to(torch.int32)
    return indices.masked_fill(score == -torch.inf, -1)


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        index_score = pytorch_extract_topk_scores(logits, topk_indices)
        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        return index_score

    @staticmethod
    def backward(ctx, grad_scores):
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
    if topk_indices is not None:
        assert not indexer_replay_manager.enabled

    weights_2d = weights.squeeze(-1)
    logits = indexer_fwd_interface(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)

    if topk_indices is None:
        topk_fn = indexer_replay_manager.get_topk_fn(
            _original_topk, return_probs=False, fill_padding_with_arange=False
        )
        topk_indices = topk_fn(logits, topk)

    index_score = IndexerFunction.apply(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, logits, topk_indices)
    return index_score, topk_indices


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
