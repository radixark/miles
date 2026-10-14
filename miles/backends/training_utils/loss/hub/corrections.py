from argparse import Namespace
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
from miles.backends.training_utils.parallel import ParallelState
from miles.utils.ft_utils.process_group_utils import GroupInfo

# Bound on log(pi_train / pi_rollout) before exponentiating.
_LOG_RATIO_LIMIT = 30.0
# Sampled-token probabilities are clamped to [eps, 1 - eps] so the binary KL stays finite.
_PROB_EPS = 1e-6


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


def binary_kl_trust_region_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    parallel_state: ParallelState,
    max_seq_lens: list[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """FlashREINFORCE sequence trust region with unclipped token IS.

    Each token is weighted by `exp(train - rollout)`. A sequence whose mean
    sampled-token binary KL, KL(Bern(mu(y_t)) || Bern(pi(y_t))) over its loss
    tokens, exceeds `--tis-binary-kl-threshold` is rejected with weight 0.
    `loss_masks` pass through unchanged: a rejected sequence still counts in the
    sample-mean denominator and keeps its entropy/KL terms. `pg_loss` is
    reweighted only under `--use-tis`; `--get-mismatch-metrics` alone just logs.
    """
    with torch.no_grad():
        train = torch.cat(train_log_probs, dim=0).float()
        rollout = torch.cat(rollout_log_probs, dim=0).float()
        log_ratio = torch.nan_to_num(train - rollout, nan=0.0, posinf=_LOG_RATIO_LIMIT, neginf=-_LOG_RATIO_LIMIT)
        ratio = log_ratio.clamp(-_LOG_RATIO_LIMIT, _LOG_RATIO_LIMIT).exp()
        binary_kl = _sampled_token_binary_kl(rollout_log_probs=rollout, train_log_probs=train)

        local_masks = get_local_response_loss_masks(
            total_lengths, response_lengths, loss_masks, args.qkv_format, max_seq_lens
        )
        seq_binary_kl = _sequence_mean(binary_kl, local_masks=local_masks, loss_masks=loss_masks, cp=parallel_state.cp)
        # NaN compares False, so a non-finite sequence is rejected.
        seq_keep = seq_binary_kl <= args.tis_binary_kl_threshold
        keep = seq_keep.repeat_interleave(torch.tensor([m.numel() for m in local_masks], device=seq_keep.device))
        weights = torch.where(keep, ratio, 0.0)

    if args.use_tis:
        pg_loss = pg_loss * weights
    metrics = {
        "tis": ratio,
        "tis_abs": (ratio - 1).abs(),
        "tis_binary_kl": binary_kl,
        "tis_seq_reject_frac": (~keep).float(),
    }
    return pg_loss, loss_masks, metrics


def _sampled_token_binary_kl(*, rollout_log_probs: torch.Tensor, train_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(Bern(p) || Bern(q)) with p = mu(y_t), q = pi(y_t): sampled token vs. the rest of the vocabulary."""
    p = rollout_log_probs.exp().clamp(_PROB_EPS, 1 - _PROB_EPS)
    q = train_log_probs.exp().clamp(_PROB_EPS, 1 - _PROB_EPS)
    return p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())


def _sequence_mean(
    values: torch.Tensor,
    *,
    local_masks: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    cp: GroupInfo,
) -> torch.Tensor:
    """Per-sequence mean of a per-token statistic over the full sequence's loss tokens.

    Under CP each rank holds a zigzag slice of every sequence, so the masked sums
    are all-reduced; every CP rank joins, including one with no local token.
    """
    sums = torch.stack(
        [
            torch.where(mask.to(device=values.device).bool(), value, 0.0).sum()
            for value, mask in zip(values.split([m.numel() for m in local_masks]), local_masks, strict=True)
        ]
    )
    if cp.size > 1:
        dist.all_reduce(sums, group=cp.group)
    counts = torch.stack([mask.sum() for mask in loss_masks]).to(sums).clamp_min(1)
    return sums / counts
