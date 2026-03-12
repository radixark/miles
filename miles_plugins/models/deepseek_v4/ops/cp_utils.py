"""
Utility functions for DeepSeek V4 Context Parallelism support.
"""

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from .ref_model import (
    get_window_topk_idxs as get_window_topk_idxs_ref,
    get_compress_topk_idxs as get_compress_topk_idxs_ref,
)


def zigzag_to_natural(tensor: Tensor, dim: int, cp_size: int) -> Tensor:
    """
    Reorder zigzag-partitioned data to natural sequential order.

    After all_gather with zigzag CP, data along `dim` is ordered as:
        [c0, c_{2cp-1}, c1, c_{2cp-2}, ...]  (by rank: rank0=[c0,c3], rank1=[c1,c2], ...)
    This function reorders to natural order:
        [c0, c1, c2, ..., c_{2cp-1}]
    """
    total = tensor.shape[dim]
    assert total % (2 * cp_size) == 0, f"dim {dim} size {total} must be divisible by 2*cp_size={2*cp_size}"
    chunk_size = total // (2 * cp_size)

    indices = list(range(0, 2 * cp_size, 2)) + list(range(2 * cp_size - 1, 0, -2))
    indices = torch.tensor(indices, device=tensor.device)

    tensor = tensor.unflatten(dim, (2 * cp_size, chunk_size))
    tensor = tensor.index_select(dim, indices)
    return tensor.flatten(dim, dim + 1)


def all_gather_cp_natural_order(
    tensor: Tensor,
    dim: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> Tensor:
    """
    All-gather tensor across CP ranks and reorder to natural sequential order.
    """
    tensor = torch.cat(torch.distributed.nn.functional.all_gather(tensor, group=cp_group), dim=dim)
    tensor = zigzag_to_natural(tensor, dim=dim, cp_size=cp_size)
    return tensor


def natural_to_zigzag_slice(tensor: Tensor, dim: int, cp_size: int, cp_rank: int) -> Tensor:
    """
    Extract zigzag slice for current CP rank from natural-ordered data.

    Zigzag pattern: rank k owns chunks [k, 2*cp_size - k - 1]
        e.g., cp_size=2: rank0=[c0,c3], rank1=[c1,c2]
    """
    total = tensor.shape[dim]
    assert total % (2 * cp_size) == 0, f"dim {dim} size {total} must be divisible by 2*cp_size={2*cp_size}"
    chunk_size = total // (2 * cp_size)

    indices = torch.tensor([cp_rank, 2 * cp_size - 1 - cp_rank], device=tensor.device)

    tensor = tensor.unflatten(dim, (2 * cp_size, chunk_size))
    tensor = tensor.index_select(dim, indices)
    return tensor.flatten(dim, dim + 1)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> Tensor:
    """
    Get global positions for local q tokens (zigzag in CP mode).
    """
    seqlen_global = seqlen_local * cp_size
    q_positions = torch.arange(0, seqlen_global, device=device)
    if cp_size > 1:
        q_positions = get_pos_emb_on_this_cp_rank(q_positions, seq_dim=0, cp_group=cp_group)
    return q_positions


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """
    Get window topk indices (CP-aware).
    """
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    base = q_positions.unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen_global, window_size), device=device)
    topk_idxs = torch.where(k_pos > base, -1, k_pos)
    result = topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = get_window_topk_idxs_ref(window_size, bsz, seqlen_local, start_pos=0)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_window_topk_idxs_cp mismatch with ref"

    return result


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """
    Get static compress topk indices (CP-aware).
    """
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    offset = seqlen_global
    k_group_idx = torch.arange(seqlen_global // ratio, device=device).repeat(seqlen_local, 1)
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    invalid_mask = k_group_idx >= q_first_invalid_group
    compress_topk_idxs = torch.where(invalid_mask, -1, k_group_idx + offset)
    result = compress_topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = get_compress_topk_idxs_ref(ratio, bsz, seqlen_local, start_pos=0, offset=offset)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_compress_topk_idxs_cp mismatch with ref"

    return result


def get_freqs_cis_for_cp(
    freqs_cis: Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> Tensor:
    """
    Get freqs_cis with proper zigzag positions for CP.
    """
    if cp_size == 1:
        return freqs_cis[:seqlen_local:stride]

    seqlen_global = seqlen_local * cp_size
    global_positions = torch.arange(0, seqlen_global, device=freqs_cis.device)
    positions = get_pos_emb_on_this_cp_rank(global_positions, seq_dim=0, cp_group=cp_group)
    return freqs_cis[positions[::stride]]
