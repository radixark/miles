"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.cp_utils``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards. Lives here rather than in one model plugin because
three of them need it (qwen3_next, qwen3_5, kimi_k3).
"""

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    Raises ``RuntimeError`` if hybrid CP is configured but ``fla.ops.cp`` is missing.
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=module.conv_kernel_size,
    )


def get_cp_sequence_lengths(cu_seqlens, cp_size, local_total_len=None):
    global_seq_lengths = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens) - 1)]
    local_seq_lengths = []
    for global_seq_len in global_seq_lengths:
        if global_seq_len % cp_size != 0:
            raise ValueError(f"Expected sequence length {global_seq_len} to be divisible by cp_size={cp_size}")
        local_seq_lengths.append(global_seq_len // cp_size)

    if local_total_len is not None and sum(local_seq_lengths) != local_total_len:
        raise ValueError(f"Expected local total length {local_total_len}, got {sum(local_seq_lengths)}")

    return global_seq_lengths, local_seq_lengths


def gather_cp_tensors(x, cp_group):
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(gathered, x.contiguous(), group=cp_group)
    return gathered


def _zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert zigzag ring-attn layout to the contiguous packed shard expected by fla CP."""
    global_seq_lengths, local_seq_lengths = get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    gathered_by_rank = [
        gathered.split(local_seq_lengths, dim=0) for gathered in gather_cp_tensors(hidden_states, cp_group)
    ]

    full_sequences = []
    for seq_idx, global_seq_len in enumerate(global_seq_lengths):
        per_rank = [rank_seqs[seq_idx] for rank_seqs in gathered_by_rank]
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            full_seq = torch.cat(
                [seq[:subchunk_len] for seq in per_rank] + [seq[subchunk_len:] for seq in per_rank][::-1],
                dim=0,
            )
        else:
            # Final local padding is appended contiguously on each rank, not in zigzag order.
            full_seq = torch.cat(per_rank, dim=0)
        full_sequences.append(full_seq)

    full_stream = torch.cat(full_sequences, dim=0) if full_sequences else hidden_states[:0]
    shard_len = hidden_states.size(0)
    return full_stream[cp_rank * shard_len : (cp_rank + 1) * shard_len]


def _packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert contiguous packed shard layout back to zigzag ring-attn layout."""
    global_seq_lengths, local_seq_lengths = get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    full_stream = torch.cat(gather_cp_tensors(hidden_states, cp_group), dim=0)
    full_sequences = full_stream.split(global_seq_lengths, dim=0)

    local_sequences = []
    for full_seq, global_seq_len, local_seq_len in zip(
        full_sequences, global_seq_lengths, local_seq_lengths, strict=True
    ):
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            parts = full_seq.split(subchunk_len, dim=0)
            local_sequences.append(torch.cat([parts[cp_rank], parts[2 * cp_size - 1 - cp_rank]], dim=0))
        else:
            local_sequences.append(full_seq.split(local_seq_len, dim=0)[cp_rank])

    return torch.cat(local_sequences, dim=0) if local_sequences else hidden_states[:0]


class _ZigzagToPackedShard(torch.autograd.Function):
    """Convert zigzag ring-attn layout to contiguous packed shards for native fla CP."""

    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return _zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = _packed_shard_to_zigzag_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


class _PackedShardToZigzag(torch.autograd.Function):
    """Convert contiguous packed shards back to zigzag ring-attn layout."""

    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return _packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = _zigzag_to_packed_shard_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _ZigzagToPackedShard.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _PackedShardToZigzag.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)
