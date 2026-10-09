"""Categorical decision-policy GRPO with exact rewards and calibration replay."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecisionGroup:
    actions: tuple[torch.Tensor, ...]
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor


def validate_targets(targets: Sequence[Sequence[float]], logits: Sequence[torch.Tensor]) -> None:
    if not logits or len(logits) != len(targets):
        raise ValueError("field/target count mismatch")
    for scores, values in zip(logits, targets, strict=True):
        target = torch.tensor(values, device=scores.device)
        if scores.ndim != 1 or scores.numel() < 2 or scores.numel() != target.numel():
            raise ValueError("invalid option dimensions")
        if not torch.isfinite(scores).all() or not torch.all((target == 0) | (target == 1)) or target.sum() != 1:
            raise ValueError("RL requires finite logits and exact single-correct-option labels")


@torch.no_grad()
def sample_group(
    logits: Sequence[torch.Tensor],
    targets: Sequence[Sequence[float]],
    group_size: int,
    generator: torch.Generator,
    record_reward_weight: float = 0.5,
) -> DecisionGroup:
    validate_targets(targets, logits)
    if group_size < 2 or not 0 <= record_reward_weight <= 1:
        raise ValueError("invalid sampling/reward configuration")
    actions, log_probs, correct = [], [], []
    for scores, target in zip(logits, targets, strict=True):
        log_p = scores.detach().float().log_softmax(-1)
        selected = torch.multinomial(log_p.exp(), group_size, replacement=True, generator=generator)
        actions.append(selected)
        log_probs.append(log_p[selected])
        correct.append(selected == torch.tensor(target, device=scores.device).argmax())
    correctness = torch.stack(correct).float()
    rewards = (1 - record_reward_weight) * correctness.mean(0) + record_reward_weight * correctness.bool().all(0).float()
    centered = rewards - rewards.mean()
    # Constant-reward groups get zero policy advantage, not numerical noise.
    advantages = centered / rewards.std(unbiased=False).clamp_min(1e-6)
    return DecisionGroup(tuple(actions), torch.stack(log_probs).sum(0), rewards, advantages)


def group_loss(
    logits: Sequence[torch.Tensor],
    targets: Sequence[Sequence[float]],
    group: DecisionGroup,
    reference: Sequence[Sequence[float]],
    *,
    clip_epsilon: float = 0.2,
    brier_weight: float = 1.0,
    kl_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    validate_targets(targets, logits)
    if not 0 < clip_epsilon < 1 or min(brier_weight, kl_weight) < 0:
        raise ValueError("invalid objective coefficients")
    if len(reference) != len(logits) or len(group.actions) != len(logits):
        raise ValueError("reference/action field count mismatch")
    if group.old_log_probs.ndim != 1 or group.old_log_probs.numel() < 2 or not torch.isfinite(group.old_log_probs).all():
        raise ValueError("invalid old-policy log probabilities")
    if group.rewards.shape != group.old_log_probs.shape or not torch.isfinite(group.rewards).all():
        raise ValueError("invalid sampled rewards")
    selected_log_probs, brier, kl = [], [], []
    entropies, collapsed = [], []
    for scores, values, selected, ref in zip(logits, targets, group.actions, reference, strict=True):
        log_p = scores.float().log_softmax(-1)
        p = log_p.exp()
        reference_p = torch.tensor(ref, device=scores.device, dtype=torch.float32)
        if reference_p.shape != p.shape or not torch.isfinite(reference_p).all() or (reference_p < 0).any() or not torch.isclose(reference_p.sum(), torch.ones((), device=p.device), atol=1e-5):
            raise ValueError("invalid frozen reference distribution")
        if selected.shape != group.old_log_probs.shape or selected.dtype != torch.long or (selected < 0).any() or (selected >= p.numel()).any():
            raise ValueError("invalid sampled actions")
        selected_log_probs.append(log_p[selected])
        brier.append((p - torch.tensor(values, device=p.device)).square().sum())
        # BF16 softmax can underflow tiny reference masses; floor only for log.
        kl.append((p * (log_p - reference_p.clamp_min(1e-30).log())).sum())
        entropies.append(-(p * log_p).sum())
        collapsed.append(p.max() >= 1 - 1e-6)
    log_ratio = torch.stack(selected_log_probs).sum(0) - group.old_log_probs.detach()
    ratio = log_ratio.exp()
    advantage = group.advantages.detach()
    if not torch.isfinite(advantage).all() or advantage.shape != ratio.shape:
        raise ValueError("invalid advantages")
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage).mean()
    calibration = torch.stack(brier).mean()
    divergence = torch.stack(kl).mean()
    loss = policy + brier_weight * calibration + kl_weight * divergence
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite RL objective")
    metrics = {
        "policy_loss": policy.detach().item(),
        "brier_loss": calibration.detach().item(),
        "reference_kl": divergence.detach().item(),
        "reward": group.rewards.mean().item(),
        "reward_std": group.rewards.std(unbiased=False).item(),
        "constant_reward_group": float(group.rewards.std(unbiased=False) == 0),
        "clip_fraction": ((ratio - 1).abs() > clip_epsilon).float().mean().item(),
        "entropy": torch.stack(entropies).mean().item(),
        "near_one_hot_fraction": torch.stack(collapsed).float().mean().item(),
    }
    correct = [scores.argmax().item() == max(range(len(values)), key=values.__getitem__) for scores, values in zip(logits, targets, strict=True)]
    metrics["field_accuracy"] = sum(correct) / len(correct)
    metrics["record_accuracy"] = float(all(correct))
    return loss, metrics
