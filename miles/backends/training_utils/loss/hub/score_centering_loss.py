"""Score centering (arXiv:2609.20807) on a microbatch's response layout, with Miles loss reduction."""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.data.sampling_mask import rollout_topk_candidate_ids
from miles.backends.training_utils.loss.hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss.hub.response_layout import response_layout
from miles.backends.training_utils.loss.hub.score_centering import (
    ScoreCenteringInputs,
    score_centering_loss,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def score_candidates(
    args: Namespace, logits: torch.Tensor, ids: torch.Tensor, *, chunk_size: int
) -> dict[str, torch.Tensor]:
    """Score ``ids`` ``[N, 1 + K]`` (sampled token, then rollout candidates) against ``logits`` ``[N, V/TP]``.

    Returns ``selected`` ``[N, 1 + K]``, plus ``entropy`` and ``kl_log_probs`` ``[N]`` when the args need them.
    """
    parallel = get_parallel_state()
    with_entropy = args.entropy_coef != 0 or args.observe_training_entropy
    replay = args.use_rollout_sampling_support_replay
    support_only = replay and not args.use_kl_loss
    selected, entropy = selected_log_probs_and_entropy(
        logits,
        ids,
        group=parallel.tp.group if parallel.tp.size > 1 else None,
        vocab_size=args.backend.vocab_size if args.train_backend == "megatron" else None,
        temperature=args.rollout_temperature,
        chunk_size=chunk_size,
        with_entropy=with_entropy and (not replay or support_only),
        sampling_support=support_only,
    )
    result = {}
    if args.use_kl_loss:
        # The reference forward scores the full vocabulary, not the replayed support.
        result["kl_log_probs"] = selected[:, 0]
    if replay and not support_only:
        # The saved candidates cover the entire realized support. Their
        # full-vocabulary logprobs share a normalizer, which cancels here.
        head_valid = ids[:, 1:] >= 0
        head = selected[:, 1:].masked_fill(~head_valid, -torch.inf)
        normalizer = torch.logsumexp(head, dim=-1, keepdim=True)
        normalizer = torch.where(head_valid.any(-1, keepdim=True), normalizer, 0.0)
        selected = selected - normalizer
        if with_entropy:
            head = selected[:, 1:].masked_fill(~head_valid, -torch.inf)
            entropy = -(head.exp() * torch.where(torch.isfinite(head), head, 0.0)).sum(-1)
    result["selected"] = selected
    if with_entropy:
        result["entropy"] = entropy
    return result


def _regularization(
    args: Namespace,
    entropy: torch.Tensor | None,
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    active: torch.Tensor,
    reduce: Callable[[torch.Tensor], torch.Tensor],
    *,
    kl_log_probs: torch.Tensor | None,
    ref_log_probs: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = log_probs.new_zeros(())
    metrics = {}
    if entropy is not None:
        entropy = reduce(torch.where(active, entropy, 0.0))
        loss = loss - args.entropy_coef * entropy
        metrics["entropy_loss"] = entropy.detach()
    if args.use_kl_loss:
        reference = ref_log_probs.detach()
        actor = log_probs if kl_log_probs is None else kl_log_probs
        # Mask before exponentiation: replacing infinity afterward can still give
        # NaN gradients through exp. Two bounded exponentials can multiply safely
        # in float32 (exp(40) * exp(40) < float32's maximum).
        valid = active & torch.isfinite(actor) & torch.isfinite(reference)
        difference = actor - reference
        valid = valid & (difference.abs() <= 40)
        ratio_log = log_probs - rollout_log_probs
        if args.use_unbiased_kl:
            valid = valid & torch.isfinite(ratio_log) & (ratio_log.abs() <= 40)
        actor = torch.where(valid, actor, 0.0)
        reference = torch.where(valid, reference, 0.0)
        # Keep the existing Miles KL estimator's gradient through the ratio.
        # This ratio uses the actual replayed sampling distribution even though
        # the reference penalty compares full-vocabulary probabilities.
        ratio = torch.where(valid, ratio_log, 0.0).exp() if args.use_unbiased_kl else None
        kl = compute_approx_kl(actor, reference, args.kl_loss_type, importance_ratio=ratio)
        kl = reduce(torch.where(valid, torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0), 0.0))
        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl
        metrics["kl_loss"] = kl.detach()
        metrics["kl_invalid_fraction"] = reduce((active & ~valid).float()).detach()
    return loss, metrics


def score_centering_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score centering on the microbatch's `ResponseLayout`: packed responses, or padded rows."""
    layout = response_layout(args, batch, logits, sum_of_sample_mean)

    def candidates(sample: int, positions: torch.Tensor) -> torch.Tensor:
        return rollout_topk_candidate_ids(batch, sample, positions)

    def head_log_probs(sample: int, positions: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(batch["rollout_topk_log_probs"][sample])[positions]

    def score(chunk: torch.Tensor, tokens: torch.Tensor, candidate_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ids = torch.cat((tokens.unsqueeze(-1), candidate_ids), dim=-1)
        return score_candidates(args, chunk, ids, chunk_size=layout.chunk_size)

    return score_centering_objective(
        args,
        scored=layout.map_logit_rows(logits, score, side=candidates, side_fill=-1),
        active=layout.loss_masks().bool(),
        rollout_log_probs=layout.rows(batch["rollout_log_probs"]),
        advantages=layout.rows(batch["advantages"]),
        candidate_ids=layout.rows_at_positions(candidates, fill=-1),
        rollout_head_log_probs=layout.rows_at_positions(head_log_probs),
        ref_log_probs=layout.rows(batch["ref_log_probs"]) if args.use_kl_loss else None,
        reduce=layout.sum_of_sample_mean,
    )


def score_centering_objective(
    args: Namespace,
    *,
    scored: dict[str, torch.Tensor],
    active: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    candidate_ids: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor | None,
    reduce: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The loss and metrics from rows laid out by the caller: per-sample responses or padded microbatch rows.

    ``scored`` comes from ``score_candidates``; row tensors are ``[N]`` and candidate tensors ``[N, K]``.
    ``reduce`` sums a ``[N]`` row value into the step loss the way the caller's layout needs.
    """
    selected = scored["selected"]
    rollout = torch.where(active, rollout_log_probs.detach(), 0.0)
    advantages = torch.where(active, advantages.detach(), 0.0)
    token_loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=selected[:, 0],
            train_head_log_probs=selected[:, 1:],
            rollout_log_probs=rollout,
            rollout_head_log_probs=rollout_head_log_probs,
            head_mask=(candidate_ids >= 0) & active.unsqueeze(-1),
            advantages=advantages,
            mode=args.score_centering_is,
            tis_clip=args.score_centering_tis_clip,
            mis_low=args.score_centering_mis_low,
            mis_high=args.score_centering_mis_high,
        ),
    )
    pg_loss = reduce(token_loss)
    loss, log = _regularization(
        args,
        scored.get("entropy"),
        selected[:, 0],
        rollout,
        active,
        reduce,
        kl_log_probs=scored.get("kl_log_probs"),
        ref_log_probs=ref_log_probs,
    )
    loss = loss + pg_loss
    log.update({key: reduce(value).detach() for key, value in metrics.as_log_dict().items()})
    log.update(loss=loss.detach(), pg_loss=pg_loss.detach())
    train_log_probs = torch.where(active, selected[:, 0].detach(), 0.0)
    log["train_rollout_logprob_abs_diff"] = reduce((train_log_probs - rollout).abs()).detach()
    # Match the policy-loss diagnostic: sampled-token k3 estimate of KL(rollout || train).
    rollout_train_kl = compute_approx_kl(rollout, train_log_probs, kl_loss_type="low_var_kl")
    rollout_train_kl = torch.where(
        active,
        torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    log["train_rollout_kl"] = reduce(rollout_train_kl).detach()
    return loss, log
