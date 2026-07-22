"""
Utility functions for DeepSeek V4 Context Parallelism support.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


@lru_cache(1)
def _get_window_topk_idxs_ref(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Reference (single-device, no-CP) window topk index builder. Used only as
    an equality oracle by :func:`get_window_topk_idxs_cp` when ``cp_size == 1``.
    """

    def _inner():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1).cuda()


@lru_cache(2)
def _get_compress_topk_idxs_ref(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """Reference (single-device, no-CP) compress topk index builder. Used only as
    an equality oracle by :func:`get_compress_topk_idxs_cp` when ``cp_size == 1``.
    """

    def _inner():
        if start_pos > 0:
            return torch.arange(0, (start_pos + 1) // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1).cuda()


def all_gather_cp(tensor: Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    """All-gather tensor across CP ranks on `dim`. Contiguous CP = result already in natural order."""
    return torch.cat(torch.distributed.nn.functional.all_gather(tensor, group=cp_group), dim=dim)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> Tensor:
    """Get global positions for local q tokens (contiguous CP)."""
    if cp_size <= 1 or cp_group is None:
        return torch.arange(0, seqlen_local, device=device)
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return torch.arange(start, start + seqlen_local, device=device)


def get_q_positions_for_packed_cp(seqlen_local: int, cp_size: int, cp_group: torch.distributed.ProcessGroup, device):
    """Return global packed-stream positions owned by this contiguous CP rank."""
    return get_q_positions_for_cp(
        seqlen_local,
        cp_size=cp_size,
        cp_group=cp_group,
        device=device,
    )


def is_packed_thd_contiguous_cp(packed_seq_params, cp_size: int) -> bool:
    """Return whether THD boundaries describe the contiguous CP layout used here."""
    return (
        packed_seq_params is not None
        and getattr(packed_seq_params, "qkv_format", None) == "thd"
        and (cp_size == 1 or bool(getattr(packed_seq_params, "miles_allgather_cp", False)))
    )


def get_seq_ids_and_offsets_from_cu_seqlens(cu_seqlens: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map packed global positions to sequence ids and in-sequence offsets."""
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must contain at least one packed sequence, got shape={tuple(cu_seqlens.shape)}")
    seq_ids = torch.searchsorted(cu_seqlens, positions.to(cu_seqlens.dtype), right=True) - 1
    seq_starts = cu_seqlens[seq_ids].to(positions.device)
    seq_ends = cu_seqlens[seq_ids + 1].to(positions.device)
    offsets = positions - seq_starts
    return seq_ids, offsets, seq_starts, seq_ends


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get window topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    base = q_positions.unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen_global, window_size), device=device)
    topk_idxs = torch.where(k_pos > base, -1, k_pos)
    result = topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = _get_window_topk_idxs_ref(window_size, bsz, seqlen_local, start_pos=0)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_window_topk_idxs_cp mismatch with ref"

    return result


def get_window_topk_idxs_packed(
    q_positions: Tensor,
    cu_seqlens: Tensor,
    *,
    window_size: int,
    bsz: int,
) -> Tensor:
    """Get local-window indices for a packed stream without crossing sequence boundaries."""
    device = q_positions.device
    _, q_offsets, seq_starts, _ = get_seq_ids_and_offsets_from_cu_seqlens(cu_seqlens, q_positions)
    width = min(int(window_size), int(cu_seqlens[-1].item()))
    rel = (q_offsets.unsqueeze(1) - width + 1).clamp(min=0) + torch.arange(width, device=device)
    k_pos = seq_starts.unsqueeze(1) + rel
    topk_idxs = torch.where(rel > q_offsets.unsqueeze(1), -1, k_pos)
    return topk_idxs.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get static compress topk indices (CP-aware)."""
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
        ref_result = _get_compress_topk_idxs_ref(ratio, bsz, seqlen_local, start_pos=0, offset=offset)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_compress_topk_idxs_cp mismatch with ref"

    return result


def get_compress_topk_idxs_packed(
    q_positions: Tensor,
    cu_seqlens: Tensor,
    *,
    ratio: int,
    bsz: int,
) -> Tensor:
    """Get compressed-KV topk indices for a packed stream without crossing boundaries."""
    device = q_positions.device
    comp_seq_starts, comp_seq_ends = get_compress_query_ranges_for_packed(q_positions, cu_seqlens, ratio=ratio)
    comp_lengths = comp_seq_ends - comp_seq_starts
    max_comp = int(comp_lengths.max().item()) if comp_lengths.numel() else 0
    if max_comp == 0:
        return torch.full((bsz, q_positions.numel(), 1), -1, device=device, dtype=torch.long)
    rel_group = torch.arange(max_comp, device=device).unsqueeze(0)
    valid = rel_group < comp_lengths.unsqueeze(1)
    comp_idx = comp_seq_starts.unsqueeze(1) + rel_group
    comp_offset = int(cu_seqlens[-1].item())
    comp_idx = torch.where(valid, comp_idx + comp_offset, -1)
    return comp_idx.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_cu_seqlens_for_packed(cu_seqlens: Tensor, *, ratio: int) -> Tensor:
    """Return compressed packed boundaries. Requires per-sample lengths divisible by ratio."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if torch.any(lengths % ratio != 0):
        raise AssertionError(f"Packed DeepSeek-V4 lengths must be divisible by compress ratio {ratio}: {lengths}")
    comp_lengths = lengths // ratio
    return torch.cat([cu_seqlens.new_zeros(1), comp_lengths.cumsum(dim=0)])


def get_compress_query_ranges_for_packed(
    q_positions: Tensor,
    cu_seqlens: Tensor,
    *,
    ratio: int,
) -> tuple[Tensor, Tensor]:
    """Return the valid compressed-KV range for each packed query token."""
    comp_cu_seqlens = get_compress_cu_seqlens_for_packed(cu_seqlens, ratio=ratio)
    seq_ids, q_offsets, _, _ = get_seq_ids_and_offsets_from_cu_seqlens(cu_seqlens, q_positions)
    starts = comp_cu_seqlens[seq_ids]
    ends = starts + (q_offsets + 1) // ratio
    ends = torch.minimum(ends, comp_cu_seqlens[seq_ids + 1])
    return starts, ends


def get_freqs_cis_for_cp(
    freqs_cis: Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> Tensor:
    """Get freqs_cis for this CP rank (contiguous slice)."""
    if cp_size == 1 or cp_group is None:
        return freqs_cis[:seqlen_local:stride]
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return freqs_cis[start : start + seqlen_local : stride]
