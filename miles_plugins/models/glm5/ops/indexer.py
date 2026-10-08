import torch

from miles_plugins.models.indexer import select_indexer_topk

from .tilelang_indexer_fwd import indexer_fwd_interface


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_backend: str = "torch",
) -> torch.Tensor:
    """Pick the keys each query attends to.

    The indexer is frozen on every path, so this returns the selection only.
    The scores it selects on carry no gradient and nothing downstream reads
    them: attention depends on the index set alone.
    """
    weights_2d = weights.squeeze(-1)
    logits = indexer_fwd_interface(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)
    return select_indexer_topk(logits, topk, backend=topk_backend)


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
