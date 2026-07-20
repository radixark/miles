"""DTensor materialization for weight export: move sharded params to CUDA, then
all-gather to full. Copied from the #1469 endstate (fsdp_utils/dtensor.py) — generic
over any DTensor placement (FSDP-shard, TP-shard, EP-shard), so it works unmodified for
torchtitan's TP/EP meshes too. Twin file: mirror fixes into fsdp_utils/dtensor.py.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


def gather_full_param(param: torch.Tensor, *, async_op: bool = False) -> torch.Tensor:
    """Materialize a (possibly sharded) param to a full local tensor on CUDA;
    async_op returns a .wait()-able."""
    full = param.cuda()
    if not isinstance(full, DTensor):
        return full
    if dist.get_world_size() == 1:
        # redistribute on a 1-rank mesh trips `assert compute_mesh is not None`
        return full.full_tensor()
    return full.redistribute(
        placements=[Replicate()] * full.device_mesh.ndim,
        async_op=async_op,
    ).to_local()
