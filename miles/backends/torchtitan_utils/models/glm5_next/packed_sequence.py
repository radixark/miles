from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd

from miles_plugins.models.cp_utils import build_fla_cp_context


@dataclass(frozen=True)
class DocumentOffsets:
    """Where each packed document starts, on the device and on the host (fla's varlen kernels take both)."""

    cu_seq_q: torch.Tensor
    cu_seq_q_host: tuple[int, ...]

    @classmethod
    def from_positions(cls, positions_T: torch.Tensor) -> "DocumentOffsets":
        starts = (positions_T == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        end = torch.tensor([positions_T.shape[0]], dtype=torch.int32, device=positions_T.device)
        cu_seq_q = torch.cat([starts, end])
        return cls(cu_seq_q=cu_seq_q, cu_seq_q_host=tuple(cu_seq_q.tolist()))


@dataclass(frozen=True)
class ContextParallelLayout:
    """This rank's contiguous ``1 / cp`` slice of the packed sequence: torchtitan shards CP inputs in order."""

    group: dist.ProcessGroup
    local_indices: torch.Tensor

    @classmethod
    def build(cls, group: dist.ProcessGroup, *, local_len: int, device) -> "ContextParallelLayout":
        start = dist.get_rank(group) * local_len
        return cls(group=group, local_indices=torch.arange(start, start + local_len, device=device))


@dataclass(frozen=True)
class PackedSequence:
    """Document offsets of the whole packed sequence and, under context parallelism, what the token-mixing
    layers need: KDA runs fla's CP on this rank's slice, DSA attends local queries over the gathered keys."""

    masks: DocumentOffsets
    cp_layout: ContextParallelLayout | None = None
    kda_cp_context: object | None = None

    @property
    def query_token_ids(self) -> torch.Tensor | None:
        return None if self.cp_layout is None else self.cp_layout.local_indices

    def gather_tokens(self, x_TD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_TD
        return all_gather_single_autograd(x_TD.contiguous(), 0, self.cp_layout.group)

    def gather_tokens_detached(self, x_TD: torch.Tensor) -> torch.Tensor:
        if self.cp_layout is None:
            return x_TD
        return _gather_tokens(x_TD, self.cp_layout.group)


def build_packed_sequence(
    positions_T: torch.Tensor, cp_group: dist.ProcessGroup | None, *, conv_kernel_size: int | None
) -> PackedSequence:
    if cp_group is None:
        return PackedSequence(masks=DocumentOffsets.from_positions(positions_T))
    cp_layout = ContextParallelLayout.build(cp_group, local_len=positions_T.shape[0], device=positions_T.device)
    with torch.no_grad():
        masks = DocumentOffsets.from_positions(_gather_tokens(positions_T, cp_group))
    return PackedSequence(
        masks=masks,
        cp_layout=cp_layout,
        kda_cp_context=(
            None
            if conv_kernel_size is None
            else build_fla_cp_context(masks.cu_seq_q, cp_group, conv_kernel_size, positions_T.device)
        ),
    )


def _gather_tokens(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = local.new_empty((dist.get_world_size(group) * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered
