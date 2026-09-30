from collections.abc import Callable, Iterable

import torch
import torch.nn as nn


def _to_pinned_host(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu", pin_memory=True).copy_(tensor, non_blocking=True)


def _to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cuda", non_blocking=True)


def _map_optimizer_state(optimizers: Iterable[torch.optim.Optimizer], fn: Callable) -> None:
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = fn(value)


@torch.no_grad()
def move_train_state(model_parts: list[nn.Module], optimizers: Iterable[torch.optim.Optimizer], device: str) -> None:
    fn = _to_pinned_host if device == "cpu" else _to_cuda
    for module in model_parts:
        module._apply(fn)
    _map_optimizer_state(optimizers, fn)
    torch.cuda.synchronize()
