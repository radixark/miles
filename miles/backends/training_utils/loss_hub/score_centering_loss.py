"""Integrate score centering with response alignment and Miles loss reduction."""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.cp_utils import (
    allgather_cp_redistribute,
    get_local_response_loss_masks,
    slice_log_prob_with_cp,
)
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.score_centering import score_centering_loss, selected_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def _candidate_log_probs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    parallel = get_parallel_state()
    result = {"selected": []}
    with_entropy = args.entropy_coef != 0 or args.observe_training_entropy
    if with_entropy:
        result["entropy"] = []
    chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    for i, (chunk, tokens, indices) in enumerate(chunks):
        ids = torch.as_tensor(batch["rollout_topk_token_ids"][i], dtype=torch.long)
        indices = torch.as_tensor(list(indices), dtype=torch.long, device=ids.device)
        ids = ids.index_select(0, indices).to(logits.device)
        ids = torch.cat((tokens.unsqueeze(-1), ids), dim=-1)
        selected, entropy = selected_log_probs_and_entropy(
            chunk,
            ids,
            group=parallel.tp.group if parallel.tp.size > 1 else None,
            vocab_size=getattr(args, "vocab_size", None),
            temperature=args.rollout_temperature,
            chunk_size=args.log_probs_chunk_size,
            with_entropy=with_entropy,
        )
        result["selected"].append(selected)
        if with_entropy:
            result["entropy"].append(entropy)
    if args.allgather_cp and parallel.cp.size > 1:
        allgather_cp_redistribute(
            result,
            logits=logits,
            args=args,
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
    return result


def _local_candidates(args: Namespace, batch: RolloutBatch, key: str, device: torch.device) -> torch.Tensor:
    values = []
    for i, (value, total, response) in enumerate(
        zip(batch[key], batch["total_lengths"], batch["response_lengths"], strict=True)
    ):
        maximum = batch["max_seq_lens"][i] if batch.get("max_seq_lens") is not None else None
        # Slice on CPU before copying to the device; each CP rank needs only its rows.
        value = slice_log_prob_with_cp(torch.as_tensor(value), total, response, args.qkv_format, maximum)
        values.append(value.to(device))
    return torch.cat(values)


def _regularization(
    args: Namespace,
    batch: RolloutBatch,
    entropy: torch.Tensor | None,
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    active: torch.Tensor,
    reduce: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = log_probs.new_zeros(())
    metrics = {}
    if entropy is not None:
        entropy = reduce(entropy)
        loss = loss - args.entropy_coef * entropy
        metrics["entropy_loss"] = entropy.detach()
    if args.use_kl_loss:
        reference = torch.where(active, torch.cat(batch["ref_log_probs"]).detach(), 0.0)
        log_probs = torch.where(active, log_probs, 0.0)
        # Keep the existing Miles KL estimator's gradient through the ratio.
        ratio = (log_probs - rollout_log_probs).exp() if args.use_unbiased_kl else None
        kl = reduce(compute_approx_kl(log_probs, reference, args.kl_loss_type, importance_ratio=ratio))
        loss = loss + args.kl_loss_coef * kl
        metrics["kl_loss"] = kl.detach()
    return loss, metrics


def score_centering_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"):
        if batch.get(key) is None:
            raise ValueError(f"Score centering requires {key} from the rollout producer")
    probabilities = _candidate_log_probs(args, batch, logits)
    selected = torch.cat(probabilities["selected"])
    active = torch.cat(
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.qkv_format,
            batch.get("max_seq_lens"),
        )
    ).bool()
    rollout = torch.where(active, torch.cat(batch["rollout_log_probs"]).detach(), 0.0)
    advantages = torch.where(active, torch.cat(batch["advantages"]).detach(), 0.0)
    ids = _local_candidates(args, batch, "rollout_topk_token_ids", logits.device)
    head = _local_candidates(args, batch, "rollout_topk_log_probs", logits.device)
    token_loss, metrics = score_centering_loss(
        selected[:, 0],
        selected[:, 1:],
        rollout,
        head,
        (ids >= 0) & active.unsqueeze(-1),
        advantages,
        center=not getattr(args, "disable_score_centering_correction", False),
        mode=args.score_centering_is,
        tis_clip=args.score_centering_tis_clip,
        mis_low=args.score_centering_mis_low,
        mis_high=args.score_centering_mis_high,
    )
    pg_loss = sum_of_sample_mean(token_loss)
    entropy = torch.cat(probabilities["entropy"]) if "entropy" in probabilities else None
    loss, log = _regularization(args, batch, entropy, selected[:, 0], rollout, active, sum_of_sample_mean)
    loss = loss + pg_loss
    log.update({key: sum_of_sample_mean(value).detach() for key, value in metrics.items()})
    log.update(loss=loss.detach(), pg_loss=pg_loss.detach())
    train_log_probs = torch.where(active, selected[:, 0].detach(), 0.0)
    log["train_rollout_logprob_abs_diff"] = sum_of_sample_mean((train_log_probs - rollout).abs()).detach()
    # Match the policy-loss diagnostic: sampled-token k3 estimate of KL(rollout || train).
    rollout_train_kl = compute_approx_kl(rollout, train_log_probs, kl_loss_type="low_var_kl")
    rollout_train_kl = torch.where(
        active,
        torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    log["train_rollout_kl"] = sum_of_sample_mean(rollout_train_kl).detach()
    return loss, log
