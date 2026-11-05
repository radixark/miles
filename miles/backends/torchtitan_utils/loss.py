from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_single_autograd
from torchtitan.components.loss import BaseLoss

from miles.backends.training_utils.loss.hub.context_parallel_logits import ContextParallelLogits


class RLLossAdapter(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config):
        self.config = config
        self._batches: list | None = None
        self._closure: Callable | None = None
        self._is_training = True
        self._results: dict[int, object] = {}
        self._cp_mesh = None
        self._grad_scale = 1.0

    def set_context_parallel(self, mesh) -> None:
        self._cp_mesh = mesh

    def set_gradient_scale(self, scale: float) -> None:
        self._grad_scale = scale

    def arm(self, batches: list, closure: Callable, *, is_training: bool) -> None:
        self._batches, self._closure, self._is_training = batches, closure, is_training
        self._results = {}

    def collect(self) -> list:
        missing = [i for i in range(len(self._batches)) if i not in self._results]
        if missing:
            raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
        return [self._results[i] for i in range(len(self._batches))]

    def _context_parallel_logits(self, pred: torch.Tensor, tokens: torch.Tensor) -> ContextParallelLogits:
        """This rank's contiguous ``1 / cp`` slice of the logits and the token each of its rows predicts.
        The loss gathers only per-token vectors, not the ``[T, V]`` logits."""
        group = self._cp_mesh.get_group()
        local_len = pred.shape[0]
        start = dist.get_rank(group) * local_len
        flat = tokens.flatten().to(pred.device)
        targets = flat.new_zeros(local_len * dist.get_world_size(group))
        targets[: flat.numel() - 1] = flat[1:]
        return ContextParallelLogits(
            local=pred.unsqueeze(0),
            targets=targets[start : start + local_len],
            gather_rows=self._gather_rows,
        )

    def _gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return all_gather_single_autograd(rows.contiguous(), 0, self._cp_mesh.get_group())

    def __call__(self, pred, target, **kwargs):
        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._cp_mesh is not None:
            pred = self._context_parallel_logits(pred, batch["tokens"])
        else:
            pred = pred.unsqueeze(0)
        if self._is_training:
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss * self._grad_scale, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
