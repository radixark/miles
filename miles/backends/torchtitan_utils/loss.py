from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.tensor import DTensor
from torchtitan.components.loss import BaseLoss


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

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        if isinstance(pred, DTensor):
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        if self._cp_mesh is not None:
            pred = all_gather_single_autograd(pred, 0, self._cp_mesh.get_group())
        pred = pred.unsqueeze(0)

        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._is_training:
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss * self._grad_scale, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
