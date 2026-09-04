"""The RL loss as torchtitan's schedule sees it.

The schedule hands its loss one (pred, target) pair per microbatch; miles'
RL loss needs the whole batch. ``RLLossAdapter`` bridges the two, and undoes
the trainer's context-parallel sharding of the logits on the way.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_gather_single_autograd
from torchtitan.components.loss import BaseLoss
from torchtitan.distributed.context_parallel import cp_shard


def _gather_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    """All-gather along dim 0 over ``mesh``, carrying gradients."""
    return all_gather_single_autograd(tensor, 0, mesh.get_group())


class RLLossAdapter(BaseLoss):
    """Trampoline between the schedule's (pred, target) and miles' RL loss.

    Targets carry the microbatch index; ``arm`` sets the batches and closure for
    the next pass. In eval mode the closure result is stashed and a zero scalar
    returned. Results are keyed by index so a repeated call for one microbatch
    (the schedule's shape-inference pass) overwrites instead of duplicating.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config=None):
        self.config = config
        self._batches: list | None = None
        self._closure: Callable | None = None
        self._mode = "train"
        self._results: dict[int, object] = {}
        self._cp_mesh = None
        self._cp_balancer = "headtail"
        self._cp_restore: dict[int, torch.Tensor] = {}

    def set_context_parallel(self, mesh, balancer_type: str) -> None:
        self._cp_mesh = mesh
        self._cp_balancer = balancer_type
        self._cp_restore = {}

    def _restore_indices(self, seq_len: int, device) -> torch.Tensor:
        """The permutation undoing the CP balancer, derived by sharding an arange through it."""
        cached = self._cp_restore.get(seq_len)
        if cached is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            (local,), _ = cp_shard(self._cp_mesh, (positions,), None, self._cp_balancer)
            order = _gather_over_mesh(local.flatten(), self._cp_mesh)
            cached = order.argsort()
            self._cp_restore[seq_len] = cached
        return cached

    def arm(self, batches: list, closure: Callable, mode: str) -> None:
        self._batches, self._closure, self._mode = batches, closure, mode
        self._results = {}

    def collect(self) -> list:
        missing = [i for i in range(len(self._batches)) if i not in self._results]
        if missing:
            raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
        return [self._results[i] for i in range(len(self._batches))]

    def _gather_context_parallel(self, pred: torch.Tensor) -> torch.Tensor:
        gathered = _gather_over_mesh(pred.transpose(0, 1).contiguous(), self._cp_mesh)
        gathered = gathered.transpose(0, 1)
        restore = self._restore_indices(gathered.shape[1], gathered.device)
        return gathered.index_select(1, restore)

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        from torch.distributed.tensor import DTensor

        if isinstance(pred, DTensor):
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        if self._cp_mesh is not None:
            pred = self._gather_context_parallel(pred)

        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._mode == "train":
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
