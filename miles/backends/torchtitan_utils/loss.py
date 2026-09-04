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
    """All-gather along dim 0 over ``mesh``, carrying gradients.

    The logit gather needs the gradient path: the loss is taken on the full
    sequence, so each rank's shard has to get its slice of the gradient back,
    and a plain all_gather yields a tensor with no autograd history that
    backward then refuses. torch's functional-collectives variant gathers along
    dim 0 only, which is why callers transpose.
    """
    return all_gather_single_autograd(tensor, 0, mesh.get_group())


class RLLossAdapter(BaseLoss):
    """Trampoline between the schedule's (pred, target) and miles' RL loss.

    Targets are microbatch-index tensors: the schedule only transports
    tensors, and the RL loss needs the whole miles batch (advantages, old log
    probs, masks), which stays outside torchtitan. ``arm`` sets the batches
    and closure for the next step; in eval mode the closure result is stashed
    and a zero scalar returned (the schedule requires a loss).

    Results are keyed by microbatch index rather than appended: the schedule
    may invoke the loss outside the scheduled microbatches (its first step
    runs a backward-metadata inference call), which upstream's pure losses
    never notice. Keying makes those calls idempotent -- the scheduled pass
    overwrites, and exactly one result per microbatch survives.
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
        """Gather CP-sharded logits before the RL loss sees them.

        Context parallelism is internal to the trainer: miles' loss hub is
        handed full-length logits, so the memory it needs is the same as at
        cp=1 while attention keeps CP's shorter sequences.
        """
        self._cp_mesh = mesh
        self._cp_balancer = balancer_type
        self._cp_restore = {}

    def _restore_indices(self, seq_len: int, device) -> torch.Tensor:
        """Where each sequence position ends up, asked of torchtitan directly.

        Rather than reproduce the load-balancing permutation, this shards a
        vector of positions through the same ``cp_shard`` the trainer shards its
        inputs with and gathers the result: slot i of the gathered logits then
        holds position ``order[i]``, and the inverse of that is the permutation
        to undo. Nothing here names a balancer, so torchtitan stays the single
        source of truth for the layout.

        Cached per length rather than computed once: without pipeline
        parallelism microbatches keep their own lengths, so one permutation
        cannot cover them all.
        """
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
        """All-gather sequence-sharded logits and undo the CP permutation.

        The gather has to carry gradients: the loss is taken on the full
        sequence, so each rank's shard needs its slice of the gradient back.
        Plain ``dist.all_gather`` produces a tensor with no autograd history
        and ``loss.backward()`` fails on it. The functional-collectives variant
        gathers along dim 0 only, hence the transpose.
        """
        gathered = _gather_over_mesh(pred.transpose(0, 1).contiguous(), self._cp_mesh)
        gathered = gathered.transpose(0, 1)
        restore = self._restore_indices(gathered.shape[1], gathered.device)
        return gathered.index_select(1, restore)

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        from torch.distributed.tensor import DTensor

        if isinstance(pred, DTensor):
            # Under TP titan shards the lm_head output over the vocab dim
            # (Shard(-1)) -- exactly the Megatron vocab-parallel dialect miles'
            # loss hub speaks (its softmax reduces over parallel_state.tp). So
            # the loss gets the local shard; gathering to full vocab instead
            # would double-count the softmax denominator, shifting every
            # log-prob by -ln(tp).
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        # After the DTensor unwrap, never before: the CP gather is a plain
        # collective over the cp mesh, and under TP the logits arrive as a
        # DTensor whose local shard is what actually has to be gathered.
        if self._cp_mesh is not None:
            pred = self._gather_context_parallel(pred)

        # Any element identifies the batch (see _microbatch_inputs); under CP
        # this rank holds only a slice of the target.
        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._mode == "train":
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
