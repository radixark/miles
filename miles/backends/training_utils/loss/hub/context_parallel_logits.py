from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContextParallelLogits:
    """Logits of this rank's context-parallel shard, handed to the loss instead of the gathered sequence.

    ``local`` is ``[1, L, V]``; row ``i`` predicts ``targets[i]``, the token at the next sequence position.
    ``gather_rows`` turns a per-row ``[L]`` vector into the sequence-ordered ``[cp * L]`` vector, with
    autograd. The log-prob consumers compute per-token values on the shard and gather only those, so no
    rank holds ``[cp * L, V]`` logits."""

    local: torch.Tensor
    targets: torch.Tensor
    gather_rows: Callable[[torch.Tensor], torch.Tensor]

    @property
    def device(self) -> torch.device:
        return self.local.device

    def sum(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.local.sum(dtype=dtype)
