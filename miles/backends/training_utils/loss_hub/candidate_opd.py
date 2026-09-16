"""Candidate-token PPO for routed on-policy distillation."""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss_hub.candidate_opd_ops import candidate_policy_loss, selected_log_softmax
from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def candidate_opd_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
):
    parallel = get_parallel_state()
    chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    losses, reverse_kls, clip_fractions, weighted_drifts, support_masses = [], [], [], [], []
    guard_fractions = []
    for index, (chunk, _) in enumerate(chunks):
        current = selected_log_softmax(
            chunk,
            batch["opd_candidate_ids"][index],
            group=parallel.tp.group,
            rank=parallel.tp.rank,
            size=parallel.tp.size,
        )
        token_loss, reverse_kl, clip_fraction = candidate_policy_loss(
            current,
            batch["opd_candidate_old_log_probs"][index],
            batch["opd_candidate_teacher_log_probs"][index],
            refresh=args.opd_reward_refresh,
            eps_low=args.eps_clip,
            eps_high=args.eps_clip_high,
            dual_clip=getattr(args, "eps_clip_c", None),
        )
        losses.append(token_loss * batch["opd_loss_weights"][index] * args.opd_kl_coef)
        reverse_kls.append(reverse_kl)
        clip_fractions.append(clip_fraction)
        old = batch["opd_candidate_old_log_probs"][index].detach()
        weighted_drifts.append((old.softmax(-1) * (current.detach() - old).abs()).sum(-1))
        support_masses.append(old.exp().sum(-1))
        guard_fractions.append(((current.detach() - old).abs() > 20).float().mean(-1))
    loss = sum_of_sample_mean(torch.cat(losses))
    return loss, dict(
        loss=loss.detach(),
        pg_loss=loss.detach(),
        opd_reverse_kl=sum_of_sample_mean(torch.cat(reverse_kls)).detach(),
        pg_clipfrac=sum_of_sample_mean(torch.cat(clip_fractions)).detach(),
        opd_old_logprob_weighted_abs_diff=sum_of_sample_mean(torch.cat(weighted_drifts)).detach(),
        opd_old_support_mass=sum_of_sample_mean(torch.cat(support_masses)).detach(),
        opd_ratio_guard_fraction=sum_of_sample_mean(torch.cat(guard_fractions)).detach(),
    )
