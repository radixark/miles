"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

from dataclasses import dataclass, fields

import torch

from miles.backends.training_utils.loss.hub.score_centering.importance_sampling import importance_sampling
from miles.backends.training_utils.loss.hub.score_centering.masks import (
    drop_inactive_nan,
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)


@dataclass(frozen=True)
class ScoreCenteringInputs:
    """Sample tensors are [tokens]; head tensors and mask are [tokens, candidates]."""

    train_log_probs: torch.Tensor
    train_head_log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor
    rollout_head_log_probs: torch.Tensor
    head_mask: torch.Tensor
    advantages: torch.Tensor
    mode: str = "none"
    tis_clip: float = 2.0
    mis_low: float = 0.5
    mis_high: float = 5.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if (
            self.train_log_probs.ndim != 1
            or self.rollout_log_probs.shape != self.train_log_probs.shape
            or self.advantages.shape != self.train_log_probs.shape
            or self.train_head_log_probs.ndim != 2
            or self.train_head_log_probs.shape[0] != self.train_log_probs.shape[0]
            or self.rollout_head_log_probs.shape != self.train_head_log_probs.shape
            or self.head_mask.shape != self.train_head_log_probs.shape
        ):
            raise ValueError("Score-centering sample tensors must be [tokens] and head tensors [tokens, candidates]")


@dataclass(frozen=True)
class ScoreCenteringMetrics:
    """Detached per-token diagnostics before the training loss reducer."""

    correction: torch.Tensor
    train_head_mass: torch.Tensor
    rollout_head_mass: torch.Tensor
    tail_ratio: torch.Tensor
    importance_weight: torch.Tensor

    def as_log_dict(self) -> dict[str, torch.Tensor]:
        """Map existing log names to the tensors without copying their storage."""
        return {f"sc_{field.name}": getattr(self, field.name) for field in fields(self)}


def score_centering_loss(inputs: ScoreCenteringInputs) -> tuple[torch.Tensor, ScoreCenteringMetrics]:
    """Return unreduced token losses and detached token metrics.

    Inputs have shape [tokens], except head tensors/mask [tokens, candidates].
    Missing head slots have a false mask. Tail masses use the appendix's floor;
    cancellation is exact for a full distribution, approximate for a true tail
    that differs from the modeled, rescaled trainer tail.
    """
    sampling = importance_sampling(
        inputs.mode, tis_clip=inputs.tis_clip, mis_low=inputs.mis_low, mis_high=inputs.mis_high
    )
    active = inputs.advantages.detach() != 0
    train_head = sanitize_head_log_probs(inputs.train_head_log_probs, inputs.head_mask, active)
    rollout_head = sanitize_head_log_probs(inputs.rollout_head_log_probs, inputs.head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, inputs.head_mask), head_probs(rollout_head, inputs.head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(inputs.eps) / (1 - p_mass).clamp_min(inputs.eps)
        alpha = sampling.tail_scale(rho, p.dtype)
        residual = sampling.head_mass(p, q, train_head - rollout_head) - alpha.unsqueeze(-1) * p
        weight = sampling.sample_weight(
            drop_inactive_nan(inputs.train_log_probs - inputs.rollout_log_probs, active, 0.0)
        )
    correction = (residual * scored_log_probs(train_head, residual)).sum(-1)
    loss = -inputs.advantages.detach() * (weight * scored_log_probs(inputs.train_log_probs, weight) - correction)
    return loss, ScoreCenteringMetrics(
        correction=correction.detach(),
        train_head_mass=p_mass,
        rollout_head_mass=q_mass,
        tail_ratio=rho,
        importance_weight=weight,
    )
