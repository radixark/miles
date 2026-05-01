import torch

from miles.utils.replay_base import indexer_replay_manager

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


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


def _flashinfer_tie_break_value() -> int:
    from sglang.srt.environ import envs

    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]


def _flashinfer_topk(logits, topk):
    import flashinfer
    from sglang.srt.environ import envs

    score, indices = flashinfer.top_k(
        logits,
        topk,
        sorted=False,
        deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
        tie_break=_flashinfer_tie_break_value(),
        dsa_graph_safe=True,
    )
    indices = indices.to(torch.int32)
    return indices.masked_fill(score == -torch.inf, -1)


def _get_topk_fn(topk_backend: str):
    if topk_backend == "torch":
        return _original_topk
    if topk_backend == "flashinfer":
        return _flashinfer_topk
    raise ValueError(f"Unsupported miles DSA topk backend: {topk_backend}")


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
    topk_backend: str = "torch",
    topk_indices: torch.Tensor | None = None,
):
    if topk_indices is not None:
        assert not indexer_replay_manager.enabled

    weights_2d = weights.squeeze(-1)
    logits = indexer_fwd_interface(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)

    if topk_indices is None:
        topk_fn = indexer_replay_manager.get_topk_fn(_get_topk_fn(topk_backend), return_probs=False)
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
