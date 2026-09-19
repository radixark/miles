"""Score-centering losses for off-policy policy-gradient training."""

from collections.abc import Callable

import torch


def score_centering_loss(
    train_log_probs: torch.Tensor,
    sampling_log_probs: torch.Tensor,
    topk_ids: torch.Tensor,
    sampled_tokens: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the centered per-token policy loss.

    ``train_log_probs`` contains the full trainer distribution. The sampler
    distribution is represented by its top-k head and the sampled-token
    log-probability; its tail is modeled as proportional to the trainer tail.
    The correction coefficients are detached so only the trainer log-probs
    receive gradients, matching the reference implementation.
    """
    if train_log_probs.ndim != 2:
        raise ValueError(f"train_log_probs must be [tokens, vocab], got {train_log_probs.shape}")
    if topk_ids.ndim != 2 or sampling_log_probs.shape != topk_ids.shape:
        raise ValueError(
            "sampling_log_probs and topk_ids must have the same [tokens, top_k] shape, "
            f"got {sampling_log_probs.shape} and {topk_ids.shape}"
        )
    if train_log_probs.size(0) != topk_ids.size(0):
        raise ValueError("trainer and sampler token counts must match")

    topk_ids = topk_ids.to(device=train_log_probs.device, dtype=torch.long)
    topk_ids = topk_ids.clamp_min(0)
    sampling_log_probs = sampling_log_probs.to(device=train_log_probs.device, dtype=torch.float32)
    sampled_tokens = sampled_tokens.to(device=train_log_probs.device, dtype=torch.long)
    sampled_log_probs = sampled_log_probs.to(device=train_log_probs.device, dtype=torch.float32)
    advantages = advantages.to(device=train_log_probs.device)

    head_log_probs = train_log_probs.gather(-1, topk_ids)
    sampled_train_log_probs = train_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    trainer_head_mass = head_log_probs.float().exp().sum(dim=-1)
    sampler_head_mass = sampling_log_probs.exp().sum(dim=-1)
    trainer_tail_mass = (1.0 - trainer_head_mass).clamp_min(eps)
    sampler_tail_mass = (1.0 - sampler_head_mass).clamp_min(eps)
    rho = sampler_tail_mass / trainer_tail_mass

    if weight_fn is None:
        head_weight = torch.ones_like(head_log_probs)
        tail_weight = torch.ones_like(rho)
        sampled_weight = torch.ones_like(sampled_train_log_probs)
    else:
        head_weight = weight_fn((head_log_probs.float() - sampling_log_probs).exp())
        tail_weight = weight_fn(rho.reciprocal())
        sampled_weight = weight_fn((sampled_train_log_probs.float() - sampled_log_probs).exp())

    alpha = rho * tail_weight
    residual = torch.exp(sampling_log_probs) * head_weight - alpha.unsqueeze(-1) * head_log_probs.float().exp()
    correction = (residual.detach() * head_log_probs.float()).sum(dim=-1)
    return -advantages.detach() * (sampled_weight.detach() * sampled_train_log_probs - correction)


def tis_weight(ratio: torch.Tensor, *, low: float = 0.0, high: float = 2.0) -> torch.Tensor:
    """Truncated importance-sampling weight used with score centering."""
    return ratio.clamp(min=low, max=high)


def mis_weight(ratio: torch.Tensor, *, low: float, high: float) -> torch.Tensor:
    """Masked importance-sampling weight used with score centering.

    MIS keeps the ratio inside the trust band and rejects tokens outside it.
    The returned mask is detached by the caller together with the centering
    coefficients, so the band only changes the estimator's weighting.
    """
    in_band = (ratio >= low) & (ratio <= high)
    return torch.where(in_band, ratio, torch.zeros_like(ratio))
