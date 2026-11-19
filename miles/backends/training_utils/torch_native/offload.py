from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from miles import platforms


def _to_pinned_host(tensor: torch.Tensor) -> torch.Tensor:
    # a non-blocking device-to-host copy is only safe into pinned memory; without it the copy must complete here
    pinned = platforms.current_platform.supports_pin_memory
    return torch.empty_like(tensor, device="cpu", pin_memory=pinned).copy_(tensor, non_blocking=pinned)


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(platforms.current_platform.current_device(), non_blocking=True)


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
