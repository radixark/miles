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


def _validate_bpo_hyperparams(bpo_smoothing: float, bpo_weight_cap: float) -> None:
    finfo = torch.finfo(torch.float32)
    if not math.isfinite(bpo_smoothing) or not finfo.tiny <= bpo_smoothing <= finfo.max:
        raise ValueError(
            f"--bpo-smoothing must be a finite positive float32 normal value "
            f"[{finfo.tiny}, {finfo.max}], got {bpo_smoothing}"
        )
    if not math.isfinite(bpo_weight_cap) or not finfo.tiny <= bpo_weight_cap <= finfo.max:
        raise ValueError(
            f"--bpo-weight-cap must be a finite positive float32 normal value "
            f"[{finfo.tiny}, {finfo.max}], got {bpo_weight_cap}"
        )


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
        _validate_bpo_hyperparams(self.bpo_smoothing, self.bpo_weight_cap)
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
    _validate_bpo_hyperparams(args.bpo_smoothing, args.bpo_weight_cap)


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
    # promote bf16/fp16 so clip thresholds do not flip and clip sums do not saturate
    calc_dtype = torch.promote_types(
        torch.promote_types(torch.float32, log_probs.dtype),
        torch.promote_types(old_log_probs.dtype, advantages.dtype),
    )
    log_probs_fp = log_probs.to(calc_dtype)
    with torch.no_grad():
        old_log_probs_fp = old_log_probs.to(calc_dtype)
        advantages_fp = advantages.to(calc_dtype)
        smoothing = torch.tensor(args.bpo_smoothing, dtype=calc_dtype, device=log_probs.device)
        numerator = smoothing - old_log_probs_fp.expm1()
        denominator = smoothing - log_probs_fp.expm1()
        # keep valid old_log_probs == -inf (q = 0) while gating non-finite log_probs or corrupted inputs
        valid = torch.isfinite(log_probs_fp) & torch.isfinite(numerator) & (numerator >= 0) & (denominator > 0)
        safe_denominator = torch.where(valid, denominator, 1.0)
        weight = torch.where(valid, numerator / safe_denominator, 1.0)
        clipped = valid & (
            ((advantages_fp > 0) & (weight > 1.0 + args.eps_clip_high))
            | ((advantages_fp < 0) & (weight < 1.0 - args.eps_clip))
        )
        coefficient = torch.where(
            valid & (~clipped),
            advantages_fp * weight.clamp(min=0.0, max=args.bpo_weight_cap),
            0.0,
        )
    # zero masked/invalid positions before multiplying so 0 * (-inf) or NaN never propagates
    active_coeff = valid & (coefficient != 0)
    pg_loss = -coefficient * torch.where(active_coeff, log_probs_fp, 0.0)
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
