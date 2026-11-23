"""Bellman Policy Optimization (BPO) custom loss plugin.

Implements the sampled-token BPO objective (https://arxiv.org/abs/2609.15987,
Eqs. 14-15) for `--loss-type custom_loss`.
"""

import math
from argparse import Namespace
from collections.abc import Callable

import torch
from pydantic import model_validator
from typing_extensions import Self

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.types import RolloutBatch


class BpoConfig(BaseConfig):
    bpo_smoothing: A[
        float,
        Arg(help="Additive smoothing epsilon for BPO complementary probabilities (1 + eps - p)."),
    ] = 0.1
    bpo_weight_cap: A[
        float,
        Arg(help="Upper bound C on the detached BPO correction weight."),
    ] = 3.0

    @model_validator(mode="after")
    def _validate_bpo_params(self) -> Self:
        if not math.isfinite(self.bpo_smoothing) or self.bpo_smoothing <= 0.0:
            raise ValueError("--bpo-smoothing must be finite and positive")
        if not math.isfinite(self.bpo_weight_cap) or self.bpo_weight_cap <= 0.0:
            raise ValueError("--bpo-weight-cap must be finite and positive")
        return self


def _validate_bpo_args(args: Namespace) -> None:
    if args.advantage_estimator != "grpo":
        raise ValueError(f"BPO requires --advantage-estimator grpo, got {args.advantage_estimator!r}")
    if args.eps_clip_c is not None:
        raise ValueError("BPO does not support Dual-clip PPO (--eps-clip-c); omit --eps-clip-c")
    if not math.isfinite(args.eps_clip) or not 0.0 <= args.eps_clip < 1.0:
        raise ValueError(f"BPO requires finite --eps-clip in [0, 1), got {args.eps_clip}")
    if args.eps_clip_high is None or not math.isfinite(args.eps_clip_high) or args.eps_clip_high < 0.0:
        raise ValueError(f"BPO requires finite, nonnegative --eps-clip-high, got {args.eps_clip_high}")
    if not math.isfinite(args.bpo_smoothing) or args.bpo_smoothing <= 0.0:
        raise ValueError(f"--bpo-smoothing must be finite and positive, got {args.bpo_smoothing}")
    if not math.isfinite(args.bpo_weight_cap) or args.bpo_weight_cap <= 0.0:
        raise ValueError(f"--bpo-weight-cap must be finite and positive, got {args.bpo_weight_cap}")


def compute_bpo_policy_loss(
    *,
    args: Namespace,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token BPO loss and clip indicator (arXiv:2609.15987, Eqs. 14-15)."""
    del ppo_kl
    # promote bf16/fp16 so weight thresholds do not flip and clip sums do not saturate at 256
    calc_dtype = torch.promote_types(log_probs.dtype, torch.float32)
    scores = log_probs.to(calc_dtype)
    with torch.no_grad():
        old_scores = old_log_probs.to(calc_dtype)
        adv = advantages.to(calc_dtype)
        weight = (args.bpo_smoothing - old_scores.expm1()) / (args.bpo_smoothing - scores.expm1())
        clipped = ((adv > 0) & (weight > 1.0 + args.eps_clip_high)) | ((adv < 0) & (weight < 1.0 - args.eps_clip))
        coefficient = adv * weight.clamp(max=args.bpo_weight_cap) * (~clipped)
    # zero masked scores before multiplying so 0 * (-inf) never produces NaN
    pg_loss = -coefficient * torch.where(coefficient != 0, scores, 0.0)
    pg_clipfrac = clipped.to(calc_dtype)
    return pg_loss, pg_clipfrac


def bpo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute BPO policy loss through `policy_loss_function`'s standard pipeline."""
    _validate_bpo_args(args)
    # keep module importable without heavy trainer/sglang deps
    from miles.backends.training_utils.loss.hub.losses import policy_loss_function

    return policy_loss_function(args, batch, logits, sum_of_sample_mean, pg_loss_fn=compute_bpo_policy_loss)


bpo_loss_function.config_class = BpoConfig
