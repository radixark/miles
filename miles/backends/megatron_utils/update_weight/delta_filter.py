"""Tensor-level delta weight filter for reducing weight transfer volume.

Between consecutive RL training steps, ~98% of bf16 weight tensors remain
bit-identical (see: https://arxiv.org/abs/2602.03839). This module caches
the previous step's HF-format tensors and filters out unchanged ones,
reducing the data sent to rollout engines by ~50x.

Usage: instantiated once per weight updater; called on each bucket of
HF-converted (name, tensor) pairs before they are sent to engines.
"""

import logging
import time

import torch

logger = logging.getLogger(__name__)


class DeltaWeightFilter:
    """Skip unchanged tensors between consecutive weight update steps.

    Maintains a CPU cache of the last-transferred value for each named
    parameter.  On each call to :meth:`filter`, tensors that are
    bitwise-identical to the cached version are dropped.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._cache: dict[str, torch.Tensor] = {}
        self._step_total = 0
        self._step_changed = 0
        self._cumulative_total = 0
        self._cumulative_changed = 0
        self._step_count = 0

    def filter(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Return only tensors whose bits changed since the last call.

        When disabled, returns *named_tensors* unchanged so callers do
        not need a separate code path.
        """
        if not self._enabled:
            return named_tensors

        filtered: list[tuple[str, torch.Tensor]] = []
        for name, tensor in named_tensors:
            self._step_total += 1
            cpu_tensor = tensor.detach().cpu()
            cached = self._cache.get(name)
            if cached is not None and cached.shape == cpu_tensor.shape and torch.equal(cached, cpu_tensor):
                continue
            filtered.append((name, tensor))
            self._cache[name] = cpu_tensor.clone()
            self._step_changed += 1

        return filtered

    def step_done(self) -> None:
        """Mark the end of one weight-update step and log sparsity."""
        if not self._enabled or self._step_total == 0:
            self._step_total = 0
            self._step_changed = 0
            return

        sparsity = 1.0 - self._step_changed / self._step_total
        logger.info(
            f"[DeltaWeight] {self._step_changed}/{self._step_total} tensors changed "
            f"(sparsity={sparsity:.2%})"
        )
        self._cumulative_total += self._step_total
        self._cumulative_changed += self._step_changed
        self._step_count += 1
        self._step_total = 0
        self._step_changed = 0
