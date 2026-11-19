"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

from bisect import bisect_right
from itertools import pairwise

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(
        module, "cp_group", None
    )  # config-access-exempt: context parallel groups are attached only to participating model modules
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _relayout_indices(cu_seqlens, cp_rank, cp_size, total):
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding). Returns ``(to_packed, to_zigzag)``: ``to_packed`` selects this
    rank's contiguous shard from the rank-concatenated zigzag shards, ``to_zigzag`` selects this rank's
    zigzag shard from the rank-concatenated contiguous shards (the packed stream itself)."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    zigzag_position = owner * shard + start // cp_size + offset % chunk + mirrored * chunk
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = position
    rows = slice(cp_rank * shard, (cp_rank + 1) * shard)
    return zigzag_position[rows], packed_position[rows]


def _gather_select(x, index, cp_group):
    gathered = x.new_empty((x.shape[0] * dist.get_world_size(group=cp_group), *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=cp_group)
    return gathered.index_select(0, index)


class _Relayout(torch.autograd.Function):
    """All-gather over CP, then select this rank's rows; the gradient is the same with the inverse map."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group):
        ctx.cp_group = cp_group
        ctx.save_for_backward(inverse_index)
        return _gather_select(x, index, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None


def _rows_before(cu_seqlens_host, position, cp_rank, cp_size):
    """Count a rank's zigzag rows before a packed-stream boundary using a host binary search."""
    if position == cu_seqlens_host[-1]:
        return position // cp_size
    sequence = bisect_right(cu_seqlens_host, position) - 1
    start, end = cu_seqlens_host[sequence : sequence + 2]
    length = end - start
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = length // (2 * cp_size) if zigzag else length // cp_size
    rows = start // cp_size + min(chunk, max(0, offset - cp_rank * chunk))
    if zigzag:
        rows += min(chunk, max(0, offset - (2 * cp_size - 1 - cp_rank) * chunk))
    return rows


def _relayout_split_sizes(cu_seqlens_host, cp_rank, cp_size, local_length):
    """Derive collective counts from Miles' existing host boundaries, without reading GPU metadata."""
    if cu_seqlens_host is None:
        raise ValueError("All-to-all CP relayout requires the existing cu_seqlens_host metadata")
    cuts = [_rows_before(cu_seqlens_host, peer * local_length, cp_rank, cp_size) for peer in range(cp_size + 1)]
    send_sizes = [end - start for start, end in pairwise(cuts)]
    recv_sizes = [
        _rows_before(cu_seqlens_host, (cp_rank + 1) * local_length, peer, cp_size)
        - _rows_before(cu_seqlens_host, cp_rank * local_length, peer, cp_size)
        for peer in range(cp_size)
    ]
    return send_sizes, recv_sizes


def _receive_permutation(to_packed, cp_size):
    """Each peer contributes a contiguous range of its monotonically ordered zigzag shard."""
    local_length = to_packed.numel()
    peers = to_packed // local_length
    rows = to_packed % local_length
    first_rows = to_packed.new_full((cp_size,), local_length)
    first_rows.scatter_reduce_(0, peers, rows, reduce="amin", include_self=True)
    counts = to_packed.new_zeros(cp_size).scatter_add_(0, peers, torch.ones_like(peers))
    offsets = counts.cumsum(0) - counts
    return offsets[peers] + rows - first_rows[peers]


def _all_to_all_select(hidden_states, index, send_sizes, recv_sizes, cp_group, to_packed):
    send_buffer = (
        hidden_states.contiguous()
        if to_packed
        else hidden_states.new_empty(hidden_states.shape).index_copy_(0, index, hidden_states)
    )
    recv_buffer = hidden_states.new_empty(hidden_states.shape)
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=cp_group,
    )
    return recv_buffer.index_select(0, index) if to_packed else recv_buffer


class _AllToAllRelayout(torch.autograd.Function):
    """A permutation's backward is its inverse transport, without CP reduction or scaling."""

    @staticmethod
    def forward(ctx, hidden_states, index, send_sizes, recv_sizes, cp_group, to_packed):
        ctx.send_sizes, ctx.recv_sizes = send_sizes, recv_sizes
        ctx.cp_group = cp_group
        ctx.to_packed = to_packed
        ctx.save_for_backward(index)
        return _all_to_all_select(hidden_states, index, send_sizes, recv_sizes, cp_group, to_packed)

    @staticmethod
    def backward(ctx, grad_output):
        (index,) = ctx.saved_tensors
        result = _all_to_all_select(
            grad_output, index, ctx.recv_sizes, ctx.send_sizes, ctx.cp_group, not ctx.to_packed
        )
        return result, None, None, None, None, None


def zigzag_to_packed_shard(
    hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, *, cu_seqlens_host=None, communication="auto"
):
    """Use A2A with native host boundaries; preserve main's AG path for legacy callers without them."""
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    if communication == "auto":
        communication = "all_to_all" if cu_seqlens_host is not None else "all_gather"
    if communication == "all_to_all":
        send_sizes, recv_sizes = _relayout_split_sizes(cu_seqlens_host, cp_rank, cp_size, hidden_states.size(0))
        index = _receive_permutation(to_packed, cp_size)
        return _AllToAllRelayout.apply(hidden_states, index, send_sizes, recv_sizes, cp_group, True)
    if communication == "all_gather":
        return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)
    raise ValueError(f"Unsupported CP communication: {communication}")


def packed_shard_to_zigzag(
    hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, *, cu_seqlens_host=None, communication="auto"
):
    """Invert the same directed transport, or retain AG when host metadata is unavailable."""
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    if communication == "auto":
        communication = "all_to_all" if cu_seqlens_host is not None else "all_gather"
    if communication == "all_to_all":
        send_sizes, recv_sizes = _relayout_split_sizes(cu_seqlens_host, cp_rank, cp_size, hidden_states.size(0))
        index = _receive_permutation(to_packed, cp_size)
        return _AllToAllRelayout.apply(hidden_states, index, recv_sizes, send_sizes, cp_group, False)
    if communication == "all_gather":
        return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)
    raise ValueError(f"Unsupported CP communication: {communication}")
