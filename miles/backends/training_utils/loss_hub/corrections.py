from argparse import Namespace
from typing import Any

import torch


def vanilla_tis_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Truncated importance sampling: clamp `exp(train - rollout)` to
    `[tis_clip_low, tis_clip]` and multiply into `pg_loss`. `loss_masks` is
    passed through unchanged; metrics report the pre-clamp ratio.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """IS clip-or-pop: zero out tokens whose `exp(train - rollout)` is outside
    `[tis_clip_low, tis_clip]` and pass the in-range ratio through unweighted.
    Same return shape as `vanilla_tis_function`.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def off_policy_is_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    cur_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Off-policy truncated IS: like `vanilla_tis_function` but with the *current*
    policy in the numerator instead of the old recompute, so the (detached) weight is
    `clip(pi_theta / pi_rollout)` against the actual rollout logprob -- one weight that
    corrects both the train/inference mismatch and async (multi-version) staleness.
    Composed with `--advantage-estimator reinforce` it is the CISPO surrogate
    (https://arxiv.org/abs/2506.13585); `--eps-clip 1.0` gives canonical single-sided
    clipping. Same `(pg_loss, loss_masks, metrics)` contract; `loss_masks` unchanged.
    """
    cur = torch.cat([lp.detach() for lp in cur_log_probs], dim=0)
    rollout = torch.cat(rollout_log_probs, dim=0)
    ratio = torch.exp(cur - rollout)
    is_weights = torch.clamp(ratio, min=1.0 - args.eps_clip, max=1.0 + args.eps_clip_high)
    is_clipfrac = (is_weights != ratio).float()
    metrics = {
        "is_weight": ratio.clone().detach(),
        "is_clipfrac": is_clipfrac.clone().detach(),
    }
    pg_loss = pg_loss * is_weights
    return pg_loss, loss_masks, metrics
