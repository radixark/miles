"""Adam moments backed by one memory-mapped file per rank, for `--fsdp-cpu-offload` runs.

The moments live in reclaimable page cache backed by node-local NVMe instead of anonymous host memory,
so the kernel writes them back rather than the pod hitting its memory limit.
"""

import logging
import os

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

_FP32_BYTES = 4


def install(optimizers: list[torch.optim.Optimizer], *, directory: str, rank: int) -> None:
    """Pre-create every param's Adam state on file-backed storage, before the first step."""
    os.makedirs(directory, exist_ok=True)
    for index, optimizer in enumerate(optimizers):
        _attach(optimizer, path=os.path.join(directory, f"rank{rank:05d}_optimizer{index}.bin"))


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _like_param(view: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    if not isinstance(param, DTensor):
        return view
    # the param's mesh is CUDA while its shard is on the host; from_local would move the moment to CUDA
    return DTensor(view, param._spec, requires_grad=False)


def _attach(optimizer: torch.optim.Optimizer, *, path: str) -> None:
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if any(optimizer.state[p] for p in params):
        raise RuntimeError(
            "--titan-optimizer-state-dir must be set up before the optimizer has state; "
            "resuming an optimizer checkpoint into file-backed moments is not supported"
        )
    if any(_local(p).device.type != "cpu" for p in params):
        raise RuntimeError("--titan-optimizer-state-dir needs the parameters on the host (--fsdp-cpu-offload)")

    numels = [_local(p).numel() for p in params]
    total = 2 * sum(numels)
    # a fresh sparse file reads as zeros, which is Adam's initial state
    with open(path, "wb") as f:
        f.truncate(total * _FP32_BYTES)
    storage = torch.from_file(path, shared=True, size=total, dtype=torch.float32)

    offset = 0
    for param, numel in zip(params, numels, strict=True):
        local_shape = _local(param).shape
        moments = []
        for _ in ("exp_avg", "exp_avg_sq"):
            moments.append(_like_param(storage[offset : offset + numel].view(local_shape), param))
            offset += numel
        state = optimizer.state[param]
        # fused AdamW expects a float32 `step` tensor on the param's device
        state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
        state["exp_avg"], state["exp_avg_sq"] = moments

    logger.info(f"Adam moments for {len(params)} params ({total * _FP32_BYTES / 1e9:.1f} GB) backed by {path}")
