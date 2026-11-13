"""Explicit original-occurrence SFT loss; targets never change tree identity."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan
from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather


@dataclass(frozen=True)
class SFTLossLayout:
    rows: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    reduction: OrderedRowGather | None = None

    @classmethod
    def from_plan(
        cls,
        plan: PrefixTreePlan,
        targets: Sequence[Sequence[int]],
        *,
        device,
        weights: Sequence[Sequence[float]] | None = None,
        deterministic_reduction=False,
    ) -> "SFTLossLayout":
        if len(targets) != len(plan.sample_nodes) or any(
            len(target) != len(nodes) for target, nodes in zip(targets, plan.sample_nodes, strict=True)
        ):
            raise ValueError("targets must contain one target per original input token; -100 ignores a token")
        if weights is not None and (
            len(weights) != len(targets) or any(len(w) != len(t) for w, t in zip(weights, targets, strict=True))
        ):
            raise ValueError("weights must contain one weight per original input token")
        rows, labels, values = [], [], []
        for sample, (nodes, target) in enumerate(zip(plan.sample_nodes, targets, strict=True)):
            for position, (row, label) in enumerate(zip(nodes, target, strict=True)):
                weight = float(weights[sample][position]) if weights is not None else 1.0
                if not 0 <= weight < float("inf"):
                    raise ValueError("SFT weights must be finite and nonnegative")
                if label != -100 and weight:
                    rows.append(row)
                    labels.append(int(label))
                    values.append(weight)
        if not rows:
            raise ValueError("SFT batch has no supervised tokens")
        return cls(
            rows=torch.tensor(rows, device=device, dtype=torch.long),
            targets=torch.tensor(labels, device=device, dtype=torch.long),
            weights=torch.tensor(values, device=device, dtype=torch.float64),
            reduction=(
                OrderedRowGather.from_indices(rows, len(plan.tokens), device=device)
                if deterministic_reduction
                else None
            ),
        )

    def loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Weighted token mean, preserving duplicate occurrences and arbitrary labels.

        Normalize over the original loss occurrences, never unique tree rows.
        Promote low-precision logits for CE, but retain float64 in the oracle.
        """
        flat = logits.reshape(-1, logits.shape[-1])
        selected = self.reduction.gather(flat) if self.reduction else flat.index_select(0, self.rows)
        if selected.dtype in (torch.float16, torch.bfloat16):
            selected = selected.float()
        losses = F.cross_entropy(selected, self.targets, reduction="none")
        weights = self.weights.to(losses.dtype)
        return (losses * weights).sum() / weights.sum()
