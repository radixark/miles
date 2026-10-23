"""Translate TorchTitan's CP shards to the contiguous streams expected by FLA."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd
from torchtitan.distributed.context_parallel import cp_shard

from miles.backends.torchtitan_utils.models.kimi_k3.ops import local_tensor, wrap_like


def _gather(x, group):
    output = x.new_empty((x.shape[0] * dist.get_world_size(group), *x.shape[1:]))
    dist.all_gather_into_tensor(output, x.contiguous(), group=group)
    return output


class _Permute(torch.autograd.Function):
    """Each token has one owner before and after the permutation, including backward."""

    @staticmethod
    def forward(ctx, x, index, inverse, group):
        ctx.group = group
        ctx.save_for_backward(inverse)
        return _gather(x, group).index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return _gather(grad, ctx.group).index_select(0, inverse), None, None, None


@dataclass(frozen=True)
class ContextParallelLayout:
    group: dist.ProcessGroup
    local_indices: torch.Tensor
    restore_indices: torch.Tensor
    contiguous_indices: torch.Tensor

    @classmethod
    def build(cls, mesh, local_length, device):
        indices = torch.arange(local_length * mesh.size(), device=device).unsqueeze(0)
        (local,), _ = cp_shard(mesh, (indices,), None, "headtail", input_seq_dim=1)
        local = local.flatten()
        restore = _gather(local, mesh.get_group()).argsort()
        start = mesh.get_local_rank() * local_length
        return cls(mesh.get_group(), local, restore, restore[start : start + local_length])

    def gather(self, x):
        """Restore global token order; autograd sums contributions from every query shard."""
        gathered = all_gather_single_autograd(x.squeeze(0).contiguous(), 0, self.group)
        return gathered.index_select(0, self.restore_indices).unsqueeze(0)

    def to_contiguous(self, x):
        output = _Permute.apply(local_tensor(x).squeeze(0), self.contiguous_indices, self.local_indices, self.group)
        return wrap_like(output.unsqueeze(0), x)

    def from_contiguous(self, x):
        output = _Permute.apply(local_tensor(x).squeeze(0), self.local_indices, self.contiguous_indices, self.group)
        return wrap_like(output.unsqueeze(0), x)
