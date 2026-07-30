"""
Utility functions for DeepSeek V4 THD (packed variable-length) support.

Indices are absolute positions into the concatenated ``[tokens | compressed]`` KV that
``deepseek_v4`` builds, so a query never reaches outside its own segment.
"""

import torch
from torch import Tensor


def batch_of_row(cu_seqlens: Tensor, total_rows: int) -> Tensor:
    """Segment index owning each row of a THD-packed tensor.

    Args:
        cu_seqlens: ``[n_seg + 1]`` cumulative lengths.
        total_rows: number of rows; rows past ``cu_seqlens[-1]`` clamp to the last segment.
    Returns:
        ``[total_rows]`` int64.
    """
    n_seg = cu_seqlens.size(0) - 1
    row_idx = torch.arange(total_rows, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def get_q_positions_thd(cu_seqlens: Tensor, total_tokens: int) -> Tensor:
    """Get positions of packed q tokens within their own segment."""
    batch_ids = batch_of_row(cu_seqlens, total_tokens)
    token_idx = torch.arange(total_tokens, device=cu_seqlens.device)
    return token_idx - cu_seqlens[batch_ids]


def get_window_topk_idxs_thd(cu_seqlens: Tensor, *, window_size: int, total_tokens: int) -> Tensor:
    """Get window topk indices for a packed stream.

    The window is clamped to the query's own segment start, so it never reaches into the
    preceding segment the way a stream-wide clamp to 0 would.

    Returns:
        ``[1, total_tokens, window_size]`` token indices, ``-1`` past the window.
    """
    device = cu_seqlens.device
    batch_ids = batch_of_row(cu_seqlens, total_tokens)
    token_idx = torch.arange(total_tokens, device=device)
    window_start = torch.maximum(token_idx - window_size + 1, cu_seqlens[batch_ids])
    k_pos = window_start.unsqueeze(1) + torch.arange(window_size, device=device)
    topk_idxs = torch.where(k_pos > token_idx.unsqueeze(1), -1, k_pos)
    return topk_idxs.unsqueeze(0)


def get_compress_topk_idxs_thd(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    ratio: int,
    total_tokens: int,
    max_n_compressed: int,
) -> Tensor:
    """Get static compress topk indices for a packed stream.

    A query sees ``(pos_in_seg + 1) // ratio`` compressed entries, clamped to the number its
    own segment produced, so segments shorter than ``ratio`` fall back to the window alone.

    Args:
        cu_seqlens_compressed: ``[n_seg + 1]`` cumulative compressed lengths from the compressor.
        max_n_compressed: column count, an upper bound on any segment's compressed length.
    Returns:
        ``[1, total_tokens, max_n_compressed]`` indices offset past the ``total_tokens``
        uncompressed rows, ``-1`` for entries the query cannot see yet.
    """
    device = cu_seqlens.device
    batch_ids = batch_of_row(cu_seqlens, total_tokens)
    token_idx = torch.arange(total_tokens, device=device)
    pos_in_seg = token_idx - cu_seqlens[batch_ids]
    seg_compressed_lens = cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]

    n_visible = ((pos_in_seg + 1) // ratio).clamp(max=seg_compressed_lens[batch_ids])
    col_idx = torch.arange(max_n_compressed, device=device).unsqueeze(0)
    compress_topk_idxs = torch.where(
        col_idx < n_visible.unsqueeze(1),
        total_tokens + cu_seqlens_compressed[batch_ids].unsqueeze(1) + col_idx,
        -1,
    )
    return compress_topk_idxs.unsqueeze(0)


def get_indexer_cu_seqlens_thd(
    cu_seqlens: Tensor, cu_seqlens_compressed: Tensor, *, ratio: int, total_tokens: int
) -> tuple[Tensor, Tensor]:
    """Get the indexer kernel's per-query KV range for a packed stream.

    Replaces the BSHD ``ks = 0`` convention, which would let a query score compressed
    entries belonging to earlier segments once all samples share one flat stream.

    Returns:
        ``(cu_ks, cu_ke)`` int32 ``[total_tokens]``, a half-open range into the compressed
        keys alone (what the indexer scores), not the concatenated KV: the query's own
        segment, up to ``(pos_in_seg + 1) // ratio`` and never past what that segment
        produced.
    """
    device = cu_seqlens.device
    batch_ids = batch_of_row(cu_seqlens, total_tokens)
    token_idx = torch.arange(total_tokens, device=device)
    pos_in_seg = token_idx - cu_seqlens[batch_ids]

    cu_ks = cu_seqlens_compressed[batch_ids]
    cu_ke = torch.minimum(cu_ks + (pos_in_seg + 1) // ratio, cu_seqlens_compressed[batch_ids + 1])
    return cu_ks.int(), cu_ke.int()
