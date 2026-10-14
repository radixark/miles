import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.tensor import DTensor
from torchtitan.components.loss import BaseLoss
from torchtitan.distributed.context_parallel import cp_shard

from miles.backends.training_utils.loss.hub.context_parallel_logits import ContextParallelLogits


def _gather_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    return all_gather_single_autograd(tensor, 0, mesh.get_group())


class RLLossAdapter(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config=None):
        self.config = config
        self._batches: list | None = None
        self._closure: Callable | None = None
        self._is_training = True
        self._results: dict[int, object] = {}
        self._cp_mesh = None
        self._cp_balancer = "headtail"
        self._cp_restore: dict[int, torch.Tensor] = {}
        self._grad_scale = 1.0

    def set_context_parallel(self, mesh, balancer_type: str) -> None:
        self._cp_mesh = mesh
        self._cp_balancer = balancer_type
        self._cp_restore = {}

    def set_gradient_scale(self, scale: float) -> None:
        self._grad_scale = scale

    def _restore_indices(self, seq_len: int, device) -> torch.Tensor:
        cached = self._cp_restore.get(seq_len)
        if cached is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            (local,), _ = cp_shard(self._cp_mesh, (positions,), None, self._cp_balancer)
            order = _gather_over_mesh(local.flatten(), self._cp_mesh)
            cached = order.argsort()
            self._cp_restore[seq_len] = cached
        return cached

    def arm(self, batches: list, closure: Callable, *, is_training: bool) -> None:
        self._batches, self._closure, self._is_training = batches, closure, is_training
        self._results = {}

    def collect(self) -> list:
        missing = [i for i in range(len(self._batches)) if i not in self._results]
        if missing:
            raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
        return [self._results[i] for i in range(len(self._batches))]

    def _context_parallel_logits(self, pred: torch.Tensor, tokens: torch.Tensor) -> ContextParallelLogits:
        """The local shard and the token each of its rows predicts. The loss gathers only per-token
        vectors, not the ``[T, V]`` logits."""
        seq_len = pred.shape[1] * self._cp_mesh.size()
        flat = tokens.flatten()
        targets = flat.new_zeros(seq_len)
        targets[: flat.numel() - 1] = flat[1:]
        (local_targets,), _ = cp_shard(self._cp_mesh, (targets.unsqueeze(0),), None, self._cp_balancer)
        restore = self._restore_indices(seq_len, pred.device)
        return ContextParallelLogits(
            local=pred,
            targets=local_targets.flatten(),
            gather_rows=functools.partial(self._gather_rows, restore=restore),
        )

    def _gather_rows(self, rows: torch.Tensor, *, restore: torch.Tensor) -> torch.Tensor:
        return _gather_over_mesh(rows.contiguous(), self._cp_mesh).index_select(0, restore)

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        if isinstance(pred, DTensor):
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._cp_mesh is not None:
            pred = self._context_parallel_logits(pred, batch["tokens"])
        if self._is_training:
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss * self._grad_scale, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
