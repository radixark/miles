"""DTensor materialization for FSDP2 weight export.

FSDP2 holds each parameter as a sharded ``DTensor``; the rollout engine needs the full tensor. Both
weight-sync paths gather it the same way -- move to the accelerator first, then all-gather to ``Replicate``.
Two caveats: ``full_tensor()`` on a CPU DTensor picks the wrong collective backend (so move to the
accelerator first), and ``redistribute`` on a 1-rank mesh trips an assert (so world_size==1 falls back to
``full_tensor()``).

Transfers that reshard themselves (tpu-sync) instead export each rank's even dim-0 block as is
(``fsdp_dim0_block``) in a buffer that owns its storage (``own_contiguous_on_device``); the FSDP and
torchtitan weight iterators share both.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from miles.platforms import current_platform


def gather_full_param(param: torch.Tensor, *, async_op: bool = False) -> torch.Tensor:
    """Materialize a (possibly FSDP2-sharded) param to a full local tensor on the accelerator.

    Non-DTensor inputs are returned moved to the accelerator unchanged. With ``async_op=True`` the all-gather is
    issued async and the returned tensor carries a ``.wait()`` the caller must drain before use.
    """
    full = param.to(current_platform.current_device())
    if not isinstance(full, DTensor):
        return full
    if dist.get_world_size() == 1:
        # redistribute on a 1-rank mesh trips `assert compute_mesh is not None`
        return full.full_tensor()
    return full.redistribute(
        placements=[Replicate()] * full.device_mesh.ndim,
        async_op=async_op,
    ).to_local()


def fsdp_dim0_block(param: torch.Tensor) -> tuple[int, int] | None:
    """``(count, index)`` if this rank holds block ``index`` of ``count`` equal dim-0 blocks of ``param``, else None.

    FSDP on a 1-D mesh shards dim 0 this way when it divides evenly. 1-D tensors, uneven cuts, 2-D meshes and
    ``_StridedShard`` (a ``Shard`` subclass that permutes the blocks) are not plain blocks.
    """
    if not isinstance(param, DTensor) or param.ndim < 2 or param.device_mesh.ndim != 1:
        return None
    (placement,) = param.placements
    if type(placement) is not Shard or placement.dim != 0:
        return None
    count = param.device_mesh.size(0)
    if count <= 1 or param.shape[0] % count or param.to_local().shape[0] != param.shape[0] // count:
        return None
    return count, param.device_mesh.get_local_rank()


def own_contiguous_on_device(tensor: torch.Tensor, device: torch.device, seen_data_ptrs: set[int]) -> torch.Tensor:
    """A detached, contiguous copy-if-needed of ``tensor`` on ``device`` that owns its storage.

    Views (``_base`` set or a nonzero storage offset) and buffers already handed out (``seen_data_ptrs``, e.g.
    tied weights) are cloned; ``seen_data_ptrs`` is updated with the returned buffer.
    """
    # checked before detach(), which clears _base on a zero-offset view
    needs_clone = tensor._base is not None or tensor.storage_offset() != 0
    tensor = tensor.detach()
    if tensor.device != device:
        tensor = tensor.to(device)
        needs_clone = False
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        needs_clone = False
    if needs_clone or tensor.data_ptr() in seen_data_ptrs:
        tensor = tensor.clone()
    seen_data_ptrs.add(tensor.data_ptr())
    return tensor
