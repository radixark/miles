from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class StepMetrics:
    grad_norm: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class StepRunner(Protocol):
    def forward_only_step(self, batches: Iterator[dict], compute: Callable[[torch.Tensor, dict], dict]) -> list: ...

    def forward_backward_step(
        self, batches: Iterator[dict], loss_closure: Callable[[torch.Tensor, dict], tuple[torch.Tensor, dict]]
    ) -> list[dict]: ...

    def zero_grad(self) -> None: ...

    def apply_step(self) -> StepMetrics: ...


def _forward_only() -> None:
    raise RuntimeError("this runner was built for forward passes only")


class LinearStepRunner:
    def __init__(
        self,
        forward_fn: Callable[[dict], torch.Tensor],
        zero_grad_fn: Callable[[], None] = _forward_only,
        step_fn: Callable[[], StepMetrics] = _forward_only,
    ):
        self._forward = forward_fn
        self._zero_grad = zero_grad_fn
        self._step = step_fn

    def forward_only_step(self, batches, compute: Callable) -> list:
        return [compute(self._forward(batch), batch) for batch in batches]

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        logs = []
        for batch in batches:
            loss, log_dict = loss_closure(self._forward(batch), batch)
            loss.backward()
            logs.append(log_dict)
        return logs

    def zero_grad(self) -> None:
        self._zero_grad()

    def apply_step(self) -> StepMetrics:
        return self._step()
