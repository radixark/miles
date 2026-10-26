from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from miles.platforms import current_platform


def _to_pinned_host(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu", pin_memory=True).copy_(tensor, non_blocking=True)


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(current_platform.current_device(), non_blocking=True)


def _map_optimizer_state(optimizers: Iterable[torch.optim.Optimizer], fn: Callable) -> None:
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = fn(value)


@torch.no_grad()
def move_train_state(
    model_parts: list[nn.Module], optimizers: Iterable[torch.optim.Optimizer], *, to_host: bool
) -> None:
    fn = _to_pinned_host if to_host else _to_device
    for module in model_parts:
        module._apply(fn)
    _map_optimizer_state(optimizers, fn)
    torch.accelerator.synchronize()
