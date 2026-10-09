from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.models.common.attention import VarlenMetadata, create_varlen_metadata_for_document


@dataclass(frozen=True)
class ContextParallelLayout:
    """Where this rank's context-parallel shard sits in the packed sequence."""

    group: dist.ProcessGroup
    local_indices: torch.Tensor
    restore_indices: torch.Tensor

    @classmethod
    def build(cls, mesh: DeviceMesh, *, load_balancer: str, seq_len: int, device) -> "ContextParallelLayout":
        sequence_order = torch.arange(seq_len, device=device).unsqueeze(0)
        (local,), _ = cp_shard(mesh, (sequence_order,), None, load_balancer)
        local_indices = local.flatten()
        gathered_order = _gather_tokens(local_indices, mesh.get_group())
        return cls(group=mesh.get_group(), local_indices=local_indices, restore_indices=gathered_order.argsort())


@dataclass(frozen=True)
class PackedSequence:
    """Document offsets of the whole packed sequence, plus the gather / keep-local pair the
    token-mixing layers (KDA, DSA) wrap themselves in under context parallelism."""

    masks: VarlenMetadata
    cp_layout: ContextParallelLayout | None = None

    def gather(self, x_BLD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_BLD
        gathered = all_gather_single_autograd(x_BLD.squeeze(0).contiguous(), 0, self.cp_layout.group)
        return gathered.index_select(0, self.cp_layout.restore_indices).unsqueeze(0)

    def keep_local(self, x_BLD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_BLD
        return x_BLD.squeeze(0).index_select(0, self.cp_layout.local_indices).unsqueeze(0)


def build_packed_sequence(positions: torch.Tensor, cp_layout: ContextParallelLayout | None) -> PackedSequence:
    if cp_layout is not None:
        with torch.no_grad():
            gathered = _gather_tokens(positions.flatten(), cp_layout.group)
            positions = gathered.index_select(0, cp_layout.restore_indices).unsqueeze(0)
    return PackedSequence(
        masks=create_varlen_metadata_for_document(positions, include_host_offsets=True),
        cp_layout=cp_layout,
    )


def _gather_tokens(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = local.new_empty((dist.get_world_size(group) * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered
