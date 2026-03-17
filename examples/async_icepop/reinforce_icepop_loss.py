"""REINFORCE + double-sided IS masking for async RL (GLM-5 Section 4.1.2).

Loss:
    L(θ) = -E[f(r, ε_l, ε_h) · Â · log π_θ]
    r = π_θ / π_rollout
    f(r) = r  if 1 - ε_l < r < 1 + ε_h
           0  otherwise

Usage:
    --loss-type custom_loss
    --custom-loss-function-path examples.reinforce_icepop_loss.reinforce_icepop_loss
    --eps-clip 0.2        # ε_l
    --eps-clip-high 0.28  # ε_h
    --advantage-estimator grpo
    --kl-coef 0
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import ParallelState
from miles.utils.types import RolloutBatch


def reinforce_icepop_loss(
    args: Namespace,
    parallel_state: ParallelState,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert not args.use_kl_loss, (
        "reinforce_icepop_loss does not support KL loss."
    )

    advantages = torch.cat(batch["advantages"], dim=0)
    rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    assert log_probs.shape == rollout_log_probs.shape, (
        f"Shape mismatch: log_probs {log_probs.shape} vs rollout_log_probs {rollout_log_probs.shape}"
    )

    # r = π_θ / π_rollout
    r = torch.exp(log_probs - rollout_log_probs)

    # f(r, ε_l, ε_h): double-sided calibration mask
    in_range = (r > 1 - args.eps_clip) & (r < 1 + args.eps_clip_high)
    f_r = torch.where(in_range, r, torch.zeros_like(r))

    # REINFORCE: -f(r) · Â · log π_θ   (f_r detached to block gradient through IS weight)
    pg_loss = sum_of_sample_mean(-(f_r.detach() * advantages * log_probs))

    # Entropy bonus
    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    # Ensure gradient flows on empty batches (e.g. some CP ranks)
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    mask_frac = sum_of_sample_mean((~in_range).float())

    train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "mask_frac": mask_frac.clone().detach(),
        "mean_r": sum_of_sample_mean(r.detach()).clone().detach(),
        "train_rollout_logprob_abs_diff": train_rollout_logprob_abs_diff.clone().detach(),
    }

    return loss, reported_loss
