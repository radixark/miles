"""Adam moments in unlinked files on node-local NVMe, for torchtitan runs under `--fsdp-cpu-offload`."""

import logging

import torch
from torch.distributed.tensor import DTensor

from miles.utils.disk_backed_tensor import disk_backed_like, purge_rank_dir

logger = logging.getLogger(__name__)

_MOMENTS = ("exp_avg", "exp_avg_sq")


def move_adam_moments_to_disk(optimizers: list[torch.optim.Optimizer], *, state_dir_root: str) -> None:
    """Back every param's Adam moments with file pages, keeping whatever a checkpoint already loaded."""
    rank_dir = purge_rank_dir(state_dir_root)
    for optimizer in optimizers:
        _move_moments_to_disk(optimizer, rank_dir)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _as_param_dtensor(local: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    if not isinstance(param, DTensor):
        return local
    # under CPU offload the param's mesh is CUDA while its shard is on the host; from_local would move it
    return DTensor(local, param._spec, requires_grad=False)


def _move_moments_to_disk(optimizer: torch.optim.Optimizer, rank_dir: str) -> None:
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if any(_local(p).device.type != "cpu" for p in params):
        raise RuntimeError(
            "--stream-optimizer-state-to-disk on torchtitan backs host-resident moments; "
            "the parameters are not on the host, pass --fsdp-cpu-offload"
        )

    numels = [_local(p).numel() for p in params]
    total = len(_MOMENTS) * sum(numels)
    buffer = disk_backed_like(torch.empty(total, dtype=torch.float32, device="meta"), rank_dir)

    offset = 0
    for param, numel in zip(params, numels, strict=True):
        state = optimizer.state[param]
        for name in _MOMENTS:
            moment = buffer[offset : offset + numel].view(_local(param).shape)
            offset += numel
            if name in state:
                moment.copy_(_local(state[name]))
            state[name] = _as_param_dtensor(moment, param)
        # fused AdamW wants a float32 `step` on the param's device
        state.setdefault("step", torch.zeros((), dtype=torch.float32))

    logger.info(f"Adam moments of {len(params)} params ({total * 4 / 1e9:.1f} GB) backed by files under {rank_dir}")
