from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import VarlenMetadata, create_varlen_metadata_for_document

from miles_plugins.models.cp_utils import build_fla_cp_context


@dataclass(frozen=True)
class ContextParallelLayout:
    """Index maps between this rank's load-balanced shard, the whole sequence and a contiguous shard.

    ``local_indices``: sequence positions of this rank's tokens. ``restore_indices``: sequence order of
    the rank-concatenated shards. ``contiguous_indices``: rows of the rank-concatenated shards that form
    this rank's contiguous ``1 / cp`` slice of the sequence."""

    group: dist.ProcessGroup
    local_indices: torch.Tensor
    restore_indices: torch.Tensor
    contiguous_indices: torch.Tensor

    @classmethod
    def build(cls, mesh: DeviceMesh, *, load_balancer: str, seq_len: int, device) -> "ContextParallelLayout":
        group = mesh.get_group()
        sequence_order = torch.arange(seq_len, device=device).unsqueeze(0)
        (local,), _ = cp_shard(mesh, (sequence_order,), None, load_balancer)
        local_indices = local.flatten()
        restore_indices = _gather_tokens(local_indices, group).argsort()
        shard_len = local_indices.shape[0]
        rank = dist.get_rank(group)
        return cls(
            group=group,
            local_indices=local_indices,
            restore_indices=restore_indices,
            contiguous_indices=restore_indices[rank * shard_len : (rank + 1) * shard_len],
        )


@dataclass(frozen=True)
class PackedSequence:
    """Document offsets of the whole packed sequence and, under context parallelism, the moves the
    token-mixing layers need: KDA runs fla's CP on a contiguous shard, DSA attends local queries
    over the gathered keys."""

    masks: VarlenMetadata
    cp_layout: ContextParallelLayout | None = None
    kda_cp_context: object | None = None

    @property
    def query_token_ids(self) -> torch.Tensor | None:
        return None if self.cp_layout is None else self.cp_layout.local_indices

    def gather_tokens(self, x_TD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_TD
        gathered = all_gather_single_autograd(x_TD.contiguous(), 0, self.cp_layout.group)
        return gathered.index_select(0, self.cp_layout.restore_indices)

    def to_contiguous(self, x_TD: torch.Tensor) -> torch.Tensor:
        layout = self.cp_layout
        return _GatherSelect.apply(x_TD, layout.contiguous_indices, layout.local_indices, layout.group)

    def from_contiguous(self, x_TD: torch.Tensor) -> torch.Tensor:
        layout = self.cp_layout
        return _GatherSelect.apply(x_TD, layout.local_indices, layout.contiguous_indices, layout.group)


def build_packed_sequence(
    positions: torch.Tensor, cp_layout: ContextParallelLayout | None, *, conv_kernel_size: int
) -> PackedSequence:
    if cp_layout is None:
        return PackedSequence(masks=create_varlen_metadata_for_document(positions, include_host_offsets=True))
    with torch.no_grad():
        gathered = _gather_tokens(positions.flatten(), cp_layout.group)
        positions = gathered.index_select(0, cp_layout.restore_indices).unsqueeze(0)
    masks = create_varlen_metadata_for_document(positions, include_host_offsets=True)
    return PackedSequence(
        masks=masks,
        cp_layout=cp_layout,
        kda_cp_context=build_fla_cp_context(masks.cu_seq_q, cp_layout.group, conv_kernel_size, positions.device),
    )


def gather_tokens_no_grad(x_TD: torch.Tensor, sequence: PackedSequence) -> torch.Tensor:
    if sequence.cp_layout is None:
        return x_TD
    return _gather_tokens(x_TD, sequence.cp_layout.group).index_select(0, sequence.cp_layout.restore_indices)


def _gather_tokens(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = local.new_empty((dist.get_world_size(group) * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered


class _GatherSelect(torch.autograd.Function):
    """A permutation of tokens across the CP group: gather every shard, keep ``index`` rows. Each
    token has one owner on either side, so the gradient is the same move with ``inverse_index``."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, group):
        ctx.group = group
        ctx.save_for_backward(inverse_index)
        return _gather_tokens(x, group).index_select(0, index)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        return _gather_tokens(grad_output, ctx.group).index_select(0, inverse_index), None, None, None
