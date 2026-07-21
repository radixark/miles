from argparse import Namespace
from collections.abc import Callable
from typing import Protocol

import torch

from miles.backends.training_utils.cp_utils import (
    all_gather_with_cp,
    get_local_response_loss_masks,
    get_sum_of_sample_mean,
)
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy, get_values
from miles.backends.training_utils.loss_hub.math_utils import (
    compute_approx_kl,
    compute_ess_ratio_contribution,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.misc import load_function
from miles.utils.types import RolloutBatch


class LossFunction(Protocol):
    """Common signature of the per-loss-type functions dispatched by `get_loss_function`."""

    def __call__(
        self,
        args: Namespace,
        batch: RolloutBatch,
        logits: torch.Tensor,
        sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute a scalar loss with gradient + a dict of detached scalar metrics.

        Args:
            args: Configuration. Each loss type reads its own subset of flags
                (PPO eps, KL coefficients, value clip, TIS settings, OPSM, etc.).
            batch: Mini-batch (`miles.utils.types.RolloutBatch`). Loss-type-
                dependent keys; common ones are `unconcat_tokens`, `response_lengths`,
                `total_lengths`, `loss_masks`, optional `max_seq_lens` (required for
                qkv_format="bshd"). Per-implementation docstrings list extras.
            logits: Float32. Last dim is vocab_size (policy/sft) or 1 (value).
                Outer shape is `[1, T, ...]` for qkv_format="thd" (T = sum of
                total_lengths) or `[B, max_seq_len, ...]` for "bshd".
            sum_of_sample_mean: CP-aware reducer; takes a flat per-token tensor
                and returns a scalar, sample-mean-weighted by `loss_masks`.

        Returns:
            `(loss, metrics)`:
              * `loss`: scalar tensor with grad, un-rescaled (the dispatcher
                applies Megatron scaling on top).
              * `metrics`: dict of detached 0-d scalars; surfaced under `train/`
                in the training log / wandb. Keys per loss type are documented
                on each implementation.
        """
        ...


# Policy-loss log keys that track pg_loss's token normalization under
# --loss-aggregation token_mean; every other policy-loss metric stays
# sample-normalized. Non-policy losses reduce every metric with the token
# reducer when the legacy per-token path is active, so their keys are
# token-normalized wholesale instead of consulting this set.
TOKEN_NORMALIZED_TRAIN_KEYS = frozenset({"loss", "pg_loss", "ess_ratio"})


def get_pg_loss_reducer(
    args: Namespace,
    batch: RolloutBatch,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    pg_loss_masks: list[torch.Tensor],
    max_seq_lens: list[int] | None,
    default_reducer: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Select the pg_loss reducer for ``--loss-aggregation``.

    ``default_reducer`` must have been built from the same ``pg_loss_masks``;
    ``sample_mean`` returns it unchanged.
    """
    mode = args.loss_aggregation
    if mode == "sample_mean":
        return default_reducer
    if mode == "token_mean":
        return get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            pg_loss_masks,
            calculate_per_token_loss=True,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
        )
    if mode == "prompt_mean":
        prompt_mask_sums = batch.get("prompt_mask_sums")
        if prompt_mask_sums is None:
            raise ValueError(
                "--loss-aggregation prompt_mean requires per-prompt-group mask sums "
                "(batch['prompt_mask_sums']), but they are missing. A custom "
                "--custom-convert-samples-to-train-data-path must populate "
                "'prompt_mask_sums' (grouped by Sample.group_index)."
            )
        reducer = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            pg_loss_masks,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
            sample_denoms=prompt_mask_sums,
        )

        def prompt_mean_reducer(x: torch.Tensor) -> torch.Tensor:
            return reducer(x) * args.n_samples_per_prompt

        return prompt_mean_reducer
    if mode == "constant":
        return get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            pg_loss_masks,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
            constant_divisor=args.loss_aggregation_divisor,
        )
    raise ValueError(f"Unknown --loss-aggregation mode: {mode!r}")


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=calculate_entropy,
        entropy_requires_grad=args.entropy_coef != 0,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    train_log_probs_list = log_probs
    old_log_probs_list = old_log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    local_loss_mask_list = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    local_loss_masks = torch.cat(local_loss_mask_list, dim=0).to(device=ppo_kl.device)
    active_tokens = local_loss_masks.bool()
    ppo_kl = torch.where(
        active_tokens,
        torch.nan_to_num(ppo_kl, nan=0.0, posinf=0.0, neginf=0.0),
        ppo_kl.new_zeros(()),
    )
    advantages = torch.where(
        active_tokens,
        torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
        advantages.new_zeros(()),
    )

    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl, advantages, args.eps_clip, args.eps_clip_high, getattr(args, "eps_clip_c", None)
    )

    if getattr(args, "dump_details", None) is not None:
        from miles.backends.training_utils.debug_dump import maybe_dump_policy_loss_debug

        maybe_dump_policy_loss_debug(
            args=args,
            batch=batch,
            train_log_probs=train_log_probs_list,
            old_log_probs=old_log_probs_list,
            rollout_log_probs=batch.get("rollout_log_probs"),
            advantages=batch["advantages"],
            local_loss_masks=local_loss_mask_list,
            ppo_kl=ppo_kl,
            pg_loss=pg_loss,
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # TIS/RS can shrink the pg_loss mask; mismatch metrics stay on the original mask.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            calculate_per_token_loss=False,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
        )

    pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]

    if args.custom_pg_loss_reducer_function_path is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = get_pg_loss_reducer(
            args,
            batch,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            pg_loss_masks=pg_loss_masks,
            max_seq_lens=max_seq_lens,
            default_reducer=sum_of_sample_mean,
        )

    # ESS (Effective Sample Size) ratio from per-token IS weights
    # w = π_new/π_old = exp(-ppo_kl).  A value of 1.0 is on-policy; near 0
    # means the per-token weights are highly concentrated.
    ess_ratio_sum = compute_ess_ratio_contribution(
        ppo_kl=ppo_kl,
        loss_masks=batch["loss_masks"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy_loss = pg_loss.new_zeros(())
    loss = pg_loss
    if calculate_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
        if args.entropy_coef != 0:
            loss = pg_loss - args.entropy_coef * entropy_loss
        else:
            entropy_loss = entropy_loss.detach()

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl = torch.where(
            active_tokens,
            torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
            kl.new_zeros(()),
        )
        kl_loss = sum_of_sample_mean(kl)

        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_scored_log_probs = old_log_probs
    train_rollout_logprob_abs_diff = None
    train_rollout_kl = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
        abs_diff = torch.where(
            active_tokens,
            torch.nan_to_num(abs_diff, nan=0.0, posinf=0.0, neginf=0.0),
            abs_diff.new_zeros(()),
        )
        train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

        # KL(rollout || train) at sampled tokens via Schulman k3 with per-token clamp [-10, 10]
        rollout_train_kl = compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
        rollout_train_kl = torch.where(
            active_tokens,
            torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
            rollout_train_kl.new_zeros(()),
        )
        train_rollout_kl = sum_of_sample_mean(rollout_train_kl)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ess_ratio": ess_ratio_sum.squeeze(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if train_rollout_kl is not None:
        reported_loss["train_rollout_kl"] = train_rollout_kl.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if batch.get("opd_reverse_kl") is not None:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def get_loss_function(args: Namespace) -> LossFunction:
    match args.loss_type:
        case "policy_loss":
            return policy_loss_function
        case "value_loss":
            return value_loss_function
        case "sft_loss":
            return sft_loss_function
        case "custom_loss":
            return load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
