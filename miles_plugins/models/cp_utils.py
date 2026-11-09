"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

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


def _all_to_all_select(hidden_states, send_order, recv_order, send_sizes, recv_sizes, cp_group):
    send_buffer = hidden_states.index_select(0, send_order)
    recv_buffer = torch.empty_like(send_buffer)
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=cp_group,
    )
    return torch.empty_like(recv_buffer).index_copy_(0, recv_order, recv_buffer)


class _AllToAllRelayout(torch.autograd.Function):
    """A permutation's backward is its inverse transport, without CP reduction or scaling."""

    @staticmethod
    def forward(ctx, hidden_states, index, inverse_index, cp_group):
        local_length = hidden_states.size(0)
        cp_size = dist.get_world_size(cp_group)
        send_peers = inverse_index // local_length
        recv_peers = index // local_length
        send_order = torch.argsort(send_peers, stable=True)
        recv_order = torch.argsort(index)
        sizes = (
            torch.stack([torch.bincount(send_peers, minlength=cp_size), torch.bincount(recv_peers, minlength=cp_size)])
            .cpu()
            .tolist()
        )
        ctx.send_sizes, ctx.recv_sizes = sizes
        ctx.cp_group = cp_group
        ctx.save_for_backward(send_order, recv_order)
        return _all_to_all_select(hidden_states, send_order, recv_order, ctx.send_sizes, ctx.recv_sizes, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        send_order, recv_order = ctx.saved_tensors
        result = _all_to_all_select(grad_output, recv_order, send_order, ctx.recv_sizes, ctx.send_sizes, ctx.cp_group)
        return result, None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, *, communication="all_to_all"):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    if communication == "all_to_all":
        return _AllToAllRelayout.apply(hidden_states, to_packed, to_zigzag, cp_group)
    if communication == "all_gather":
        return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)
    raise ValueError(f"Unsupported CP communication: {communication}")


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, *, communication="all_to_all"):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    if communication == "all_to_all":
        return _AllToAllRelayout.apply(hidden_states, to_zigzag, to_packed, cp_group)
    if communication == "all_gather":
        return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)
    raise ValueError(f"Unsupported CP communication: {communication}")
