from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from miles.utils.types import RolloutBatch


@dataclass(frozen=True)
class LossFnInput:
    args: Namespace
    batch: RolloutBatch
    logits: torch.Tensor
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LossFnOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class LossFunction(Protocol):
    """Common signature of the per-loss-type functions dispatched by `get_loss_function`.

    A loss function consumes a `LossFnInput` (args, batch, logits, CP-aware
    per-sample mean reducer) and returns a `LossFnOutput` with the scalar
    loss tensor (with gradient) plus a dict of detached scalar metrics for
    logging.
    """

    def __call__(self, input: LossFnInput) -> LossFnOutput: ...
