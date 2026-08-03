"""
Utility functions for DeepSeek V4 THD (packed variable-length) support.

Indices are absolute positions into the concatenated ``[tokens | compressed]`` KV that
``deepseek_v4`` builds, so a query never reaches outside its own segment.

Under context parallelism a rank holds ``[global_start, global_start + total_tokens)`` of a
globally-numbered ``cu_seqlens``, while ``deepseek_v4`` all-gathers the KV. Passing
``global_start`` resolves local rows against the global stream; the returned indices stay
absolute, so the KV layout is unchanged.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class ThdLayout:
    """How this rank's packed stream is laid out; ``None`` stands for the unpacked one.

    The first three fields come from the packed sequence parameters. The rest are filled in as
    the forward runs: the compaction ones only under CP, where a compressed group can straddle
    the split, and ``cu_seqlens_compressed`` once the compressor has grouped the stream.
    """

    cu_seqlens: Tensor
    global_start: int
    max_seqlen: int
    hidden_compact: Tensor | None = None
    compressed_group_ids: Tensor | None = None
    seq_to_rank_row: Tensor | None = None
    cu_seqlens_compressed: Tensor | None = None

    @classmethod
    def from_packed_seq_params(cls, packed_seq_params, *, cp_rank: int, seqlen_local: int):
        """This rank's layout, or None for any format other than thd."""
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            return None
        return cls(
            cu_seqlens=packed_seq_params.cu_seqlens_q,
            # CP splits the packed stream contiguously, so this rank's rows start here globally.
            global_start=cp_rank * seqlen_local,
            max_seqlen=packed_seq_params.max_seqlen_q,
        )


def batch_of_row(cu_seqlens: Tensor, total_rows: int, global_start: int = 0) -> Tensor:
    """Segment index owning each row of a THD-packed tensor.

    Args:
        cu_seqlens: ``[n_seg + 1]`` cumulative lengths.
        total_rows: number of rows; rows past ``cu_seqlens[-1]`` clamp to the last segment.
        global_start: first global row this rank holds.
    Returns:
        ``[total_rows]`` int64.
    """
    n_seg = cu_seqlens.size(0) - 1
    row_idx = torch.arange(total_rows, device=cu_seqlens.device, dtype=torch.int64) + global_start
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def compressed_cu_seqlens(cu_seqlens: Tensor, ratio: int) -> Tensor:
    """Cumulative compressed lengths, flooring each segment's tail.

    A pure function of ``cu_seqlens``, so the CP path can build it without the compressor,
    which returns None once its input is pre-grouped.
    """
    lens = torch.div(cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor")
    return torch.cat([torch.zeros_like(cu_seqlens[:1]), lens.cumsum(0).to(cu_seqlens.dtype)])


def get_q_positions_thd(cu_seqlens: Tensor, total_tokens: int, global_start: int = 0) -> Tensor:
    """Get positions of packed q tokens within their own segment."""
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=cu_seqlens.device) + global_start
    return token_idx - cu_seqlens[batch_ids]


def get_window_topk_idxs_thd(
    cu_seqlens: Tensor, *, window_size: int, total_tokens: int, global_start: int = 0
) -> Tensor:
    """Get window topk indices for a packed stream.

    The window is clamped to the query's own segment start, so it never reaches into the
    preceding segment the way a stream-wide clamp to 0 would. Under CP the window may reach
    rows another rank produced, which the KV all-gather has already delivered.

    Returns:
        ``[1, total_tokens, window_size]`` token indices, ``-1`` past the window.
    """
    device = cu_seqlens.device
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=device) + global_start
    window_start = torch.maximum(token_idx - window_size + 1, cu_seqlens[batch_ids])
    k_pos = window_start.unsqueeze(1) + torch.arange(window_size, device=device)
    topk_idxs = torch.where(k_pos > token_idx.unsqueeze(1), -1, k_pos)
    return topk_idxs.unsqueeze(0)


def to_rank_major_rows(idxs: Tensor, seq_to_rank_row: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Translate sequence-major compressed ids to their rows in the all-gathered buffer.

    Both the rule-based indices and the indexer's top-k are produced in sequence-major order,
    while CP all-gathers fixed-capacity per-rank blocks; ``-1`` drops a row no rank produced.

    ``valid`` rules a lane out only after the gather, so both ends are clamped first: a segment
    shorter than the longest one never fills its columns, and the indexer leaves its unused
    top-k slots at whatever the kernel wrote. A lane ``valid`` keeps already addresses a row
    the table holds, so the clamp moves no live index.
    """
    n_rows = seq_to_rank_row.numel()
    if n_rows == 0:
        # The whole stream is shorter than the ratio, so no rank produced a compressed row.
        # Clamping would ask for row -1 here; the empty table has no row to fall back on.
        return torch.full_like(idxs, -1), torch.zeros_like(valid)
    rows = seq_to_rank_row[idxs.clamp(0, n_rows - 1).long()]
    return rows, valid & (rows >= 0)


def get_compress_topk_idxs_thd(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    ratio: int,
    total_tokens: int,
    max_n_compressed: int,
    kv_offset: int | None = None,
    global_start: int = 0,
    seq_to_rank_row: Tensor | None = None,
) -> Tensor:
    """Get static compress topk indices for a packed stream.

    A query sees ``(pos_in_seg + 1) // ratio`` compressed entries, clamped to the number its
    own segment produced, so segments shorter than ``ratio`` fall back to the window alone.

    Args:
        cu_seqlens_compressed: ``[n_seg + 1]`` cumulative compressed lengths from the compressor.
        max_n_compressed: column count, an upper bound on any segment's compressed length.
        kv_offset: rows the compressed block starts after; defaults to ``total_tokens``, which
            only holds without CP, where the rank owns the whole stream.
        seq_to_rank_row: sequence-major to all-gather row map, which CP pads to a fixed per-rank
            capacity; ``-1`` drops a row no rank produced.
    Returns:
        ``[1, total_tokens, max_n_compressed]`` indices offset past the uncompressed rows,
        ``-1`` for entries the query cannot see yet.
    """
    device = cu_seqlens.device
    if kv_offset is None:
        kv_offset = total_tokens
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=device) + global_start
    pos_in_seg = token_idx - cu_seqlens[batch_ids]
    seg_compressed_lens = cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]

    n_visible = ((pos_in_seg + 1) // ratio).clamp(max=seg_compressed_lens[batch_ids])
    col_idx = torch.arange(max_n_compressed, device=device).unsqueeze(0)
    seq_major_idx = cu_seqlens_compressed[batch_ids].unsqueeze(1) + col_idx
    visible = col_idx < n_visible.unsqueeze(1)
    if seq_to_rank_row is not None:
        seq_major_idx, visible = to_rank_major_rows(seq_major_idx, seq_to_rank_row, visible)
    compress_topk_idxs = torch.where(visible, kv_offset + seq_major_idx, -1)
    return compress_topk_idxs.unsqueeze(0)


def get_indexer_cu_seqlens_thd(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    ratio: int,
    total_tokens: int,
    global_start: int = 0,
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
    batch_ids = batch_of_row(cu_seqlens, total_tokens, global_start)
    token_idx = torch.arange(total_tokens, device=device) + global_start
    pos_in_seg = token_idx - cu_seqlens[batch_ids]

    cu_ks = cu_seqlens_compressed[batch_ids]
    cu_ke = torch.minimum(cu_ks + (pos_in_seg + 1) // ratio, cu_seqlens_compressed[batch_ids + 1])
    return cu_ks.int(), cu_ke.int()


# --------------------------------------------------------------------------------------
# Context-parallel compressor input
#
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/95e4bafebaa799d166975ef82066a3c46648e004/megatron/core/transformer/experimental_attention_variant/csa_cp_layout_kernels.py
# --------------------------------------------------------------------------------------


def compressor_boundary_width(ratio: int) -> int:
    """Hidden rows a rank needs from its left CP neighbour.

    A group straddling the split needs up to ``ratio - 1`` rows the previous rank owns. The
    ratio=4 layers additionally overlap-transform against the preceding group, which
    ``torch.roll`` reads from the row before, so the compact buffer must start one whole
    group early; the wider window is what guarantees that.
    """
    return 8 if ratio == 4 else ratio


def compact_group_capacity(l_local: int, ratio: int) -> int:
    """Fixed per-rank compressed-group count, since all-gather needs equal sends.

    A rank's buffer spans ``l_local + d_comp`` global positions, so that many tokens bound the
    groups it can hold. mcore rounds this up for its CuTe tile alignment; the gather here takes
    any length, and every rank derives the same count from the same inputs.
    """
    return max(1, (l_local + compressor_boundary_width(ratio)) // ratio)


def _first_visible_group(range_start: Tensor | int, seg_start: Tensor, ratio: int) -> Tensor:
    """First compressed group a rank starting at ``range_start`` puts in its compact buffer.

    Shared by the compaction and the row map so the slot a group is written to and the slot it
    is looked up in cannot drift apart.
    """
    d_comp = compressor_boundary_width(ratio)
    return torch.div((range_start - d_comp - seg_start).clamp(min=0) + ratio - 1, ratio, rounding_mode="floor")


def compact_gather_index(
    cu_seqlens: Tensor, *, global_start: int, l_local: int, ratio: int, c_cap: int
) -> tuple[Tensor, Tensor]:
    """Source rows feeding each row of a rank's compact compressor input.

    A compressed group belongs to the rank holding its last token, and the group's own tokens
    may start on the previous rank, so sources address the concatenated
    ``[boundary(d_comp) | local(l_local)]`` rows rather than the local rows alone. Trailing
    ``seqlen % ratio`` tokens get no group, matching the non-CP path.

    Returns:
        ``(gather_idx [c_cap * ratio], comp_ids [c_cap])``, both ``-1`` past the groups this
        rank actually produces, which the fixed capacity over-allocates for.
    """
    device = cu_seqlens.device
    d_comp = compressor_boundary_width(ratio)
    n_seg = cu_seqlens.size(0) - 1
    range_start, range_end = global_start, global_start + l_local

    seg_start = cu_seqlens[:-1]
    seg_stop = torch.minimum(cu_seqlens[1:], torch.full_like(cu_seqlens[1:], range_end))
    first_group = _first_visible_group(range_start, seg_start, ratio)
    stop_group = torch.div(seg_stop - seg_start, ratio, rounding_mode="floor")
    n_groups = (stop_group - first_group).clamp(min=0)
    n_groups = torch.where((seg_start < seg_stop) & (range_start < seg_stop), n_groups, 0)

    cu_groups = torch.cat([torch.zeros_like(n_groups[:1]), n_groups.cumsum(0)])
    slot = torch.arange(c_cap, device=device, dtype=cu_seqlens.dtype)
    seg_ids = torch.bucketize(slot, cu_groups[1:], right=True).clamp(max=max(n_seg - 1, 0))
    valid = slot < cu_groups[-1]

    comp_ids = first_group[seg_ids] + (slot - cu_groups[seg_ids])
    group_head = seg_start[seg_ids] + comp_ids * ratio
    # The boundary rows precede the local ones, so both sides of the split share one offset.
    gather_idx = (
        group_head.unsqueeze(1) + torch.arange(ratio, device=device, dtype=cu_seqlens.dtype) - (range_start - d_comp)
    )
    gather_idx = torch.where(valid.unsqueeze(1), gather_idx, -1).flatten()
    return gather_idx.long(), torch.where(valid, comp_ids, -1).int()


class CompressorInputCompact(torch.autograd.Function):
    """Gather each visible compressed group's source tokens into a fixed-capacity buffer.

    Replaces mcore's CuTe kernel, which exists to keep shapes host-known and CUDA-graph
    capturable rather than for the arithmetic: this is integer indexing plus one gather.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_local: Tensor,
        boundary: Tensor,
        cu_seqlens: Tensor,
        global_start: int,
        ratio: int,
        c_cap: int,
    ):
        gather_idx, comp_ids = compact_gather_index(
            cu_seqlens,
            global_start=global_start,
            l_local=hidden_local.size(0),
            ratio=ratio,
            c_cap=c_cap,
        )
        keep = (gather_idx >= 0).view(-1, *([1] * (hidden_local.dim() - 1)))
        source = torch.cat([boundary, hidden_local], dim=0)
        compact = source.index_select(0, gather_idx.clamp(min=0)) * keep
        ctx.save_for_backward(gather_idx, keep)
        ctx.row_split = [boundary.size(0), hidden_local.size(0)]
        return compact, comp_ids

    @staticmethod
    def backward(ctx, grad_compact: Tensor, _grad_comp_ids: Tensor):
        gather_idx, keep = ctx.saved_tensors
        # A token belongs to exactly one group, so gather_idx is injective where it is valid
        # and index_add_ never accumulates; keep stops capacity padding reaching row 0.
        grad_source = grad_compact.new_zeros((sum(ctx.row_split),) + grad_compact.shape[1:])
        grad_source.index_add_(0, gather_idx.clamp(min=0), grad_compact * keep)
        grad_boundary, grad_local = grad_source.split(ctx.row_split, dim=0)
        return grad_local, grad_boundary, None, None, None, None


# --------------------------------------------------------------------------------------
# Context-parallel row map and boundary exchange
#
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/95e4bafebaa799d166975ef82066a3c46648e004/megatron/core/transformer/experimental_attention_variant/csa_cp_utils.py
# --------------------------------------------------------------------------------------


class _LeftBoundaryExchange(torch.autograd.Function):
    """Receive the left CP neighbour's last ``d_comp`` rows."""

    @staticmethod
    def forward(ctx, tensor: Tensor, d_comp: int, cp_group) -> Tensor:
        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        if tensor.size(0) < d_comp:
            raise RuntimeError(
                f"DSv4 CP boundary exchange needs at least {d_comp} local rows, " f"got {tensor.size(0)}"
            )
        ctx.input_shape, ctx.d_comp, ctx.cp_group = tensor.shape, d_comp, cp_group
        boundary = tensor.new_zeros((d_comp,) + tuple(tensor.shape[1:]))
        ops = []
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group))
        if cp_rank + 1 < cp_size:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    tensor[-d_comp:].contiguous(),
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: Tensor):
        cp_group = ctx.cp_group
        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        # The rows were read from the left neighbour, so their gradient goes back the same way.
        received = grad_boundary.new_zeros((ctx.d_comp,) + tuple(ctx.input_shape[1:]))
        ops = []
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.irecv, received, dist.get_global_rank(cp_group, cp_rank + 1), cp_group))
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_boundary.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-ctx.d_comp :] = received
        return grad_input, None, None


def exchange_cp_boundary_hidden(hidden: Tensor, *, ratio: int, cp_group) -> Tensor:
    """Boundary hidden rows the compressor needs from the left CP neighbour.

    Only the compressor needs them: the attention window reads rows the KV all-gather has
    already delivered, so the width is ``d_comp`` rather than mcore's
    ``max(csa_window_size, d_comp)``.
    """
    d_comp = compressor_boundary_width(ratio)
    flat = hidden.reshape(hidden.size(0), -1)
    boundary = _LeftBoundaryExchange.apply(flat, d_comp, cp_group)
    return boundary.reshape((d_comp,) + tuple(hidden.shape[1:]))


def compressed_rank_layout(
    cu_seqlens: Tensor,
    cu_seqlens_compressed: Tensor,
    *,
    l_local: int,
    cp_size: int,
    ratio: int,
    c_cap: int,
) -> Tensor:
    """Map sequence-major compressed rows to their rows in the all-gathered buffer.

    All-gather concatenates fixed-capacity per-rank blocks, so a group's sequence-major id is
    not its row: rank ``r`` owns slots ``[r * c_cap, (r + 1) * c_cap)`` and the unused tail of
    each block is padding. A group belongs to the rank holding its last token.

    Returns:
        ``[(l_local * cp_size) // ratio]`` int32, ``-1`` for a row no rank produced.
    """
    device = cu_seqlens.device
    n_seg = cu_seqlens.size(0) - 1
    logical_rows = torch.arange((l_local * cp_size) // ratio, device=device, dtype=cu_seqlens.dtype)
    seg_ids = torch.bucketize(logical_rows, cu_seqlens_compressed[1:], right=True).clamp(max=max(n_seg - 1, 0))
    comp_ids = logical_rows - cu_seqlens_compressed[seg_ids]
    group_last = cu_seqlens[seg_ids] + (comp_ids + 1) * ratio - 1
    owner = torch.div(group_last, l_local, rounding_mode="floor").clamp(0, cp_size - 1)

    rank_starts = torch.arange(cp_size, device=device, dtype=cu_seqlens.dtype) * l_local
    first_seg = torch.bucketize(rank_starts, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))
    first_logical = cu_seqlens_compressed[first_seg] + _first_visible_group(rank_starts, cu_seqlens[first_seg], ratio)

    rank_rows = owner * c_cap + (logical_rows - first_logical[owner])
    return torch.where(logical_rows < cu_seqlens_compressed[-1], rank_rows, -1).int()
