from argparse import Namespace
from collections.abc import Callable, Iterator

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.utils.types import RolloutBatch


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of old-student log-probability tensors. OPD
            treats these as fixed scoring inputs.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None or getattr(args, "opd_divergence", "reverse_kl") == "forward_kl":
        return

    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        if len(advantages) != len(precomputed_reverse_kls):
            raise ValueError(
                f"OPD length mismatch: advantages={len(advantages)}, "
                f"opd_reverse_kl={len(precomputed_reverse_kls)}."
            )

        reverse_kls = []
        for i, adv in enumerate(advantages):
            reverse_kl = precomputed_reverse_kls[i]
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            # Defensive consumer boundary for direct callers that bypass
            # compute_advantages_and_returns' persistent-data detach.
            reverse_kl = reverse_kl.detach().to(device=adv.device)
            if adv.shape != reverse_kl.shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"opd_reverse_kl={tuple(reverse_kl.shape)}."
                )
            advantages[i] = adv - args.opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)

        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"OPD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )

    device = student_log_probs[0].device
    detached_teacher_log_probs = [t.detach() for t in teacher_log_probs]
    rollout_data["teacher_log_probs"] = detached_teacher_log_probs
    teacher_log_probs = [t.to(device=device) for t in detached_teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        if student_log_probs[i].shape != teacher_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: student_log_probs={tuple(student_log_probs[i].shape)}, "
                f"teacher_log_probs={tuple(teacher_log_probs[i].shape)}."
            )
        if adv.shape != student_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                f"student_log_probs={tuple(student_log_probs[i].shape)}. "
                "OPD expects per-token advantages; broadcast scalar advantages must be expanded before this call."
            )
        old_student_log_prob = student_log_probs[i].detach()
        reverse_kl = old_student_log_prob - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging.
    rollout_data["opd_reverse_kl"] = reverse_kls


def iter_forward_kl_terms(
    args: Namespace, batch: RolloutBatch, logits: torch.Tensor
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield per-entry KL(teacher || student) contributions and teacher probabilities."""
    metadata = batch.get("metadata")
    if metadata is None or len(metadata) != len(batch["response_lengths"]):
        raise ValueError("Forward KL requires teacher support in batch metadata.")
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )
    for (student_logits, _), entry, length in zip(responses, metadata, batch["response_lengths"], strict=True):
        support = (entry or {}).get("opd")
        if support is None:
            raise ValueError("Forward KL requires metadata['opd'] for every sample.")
        ids = torch.as_tensor(support["ids"], device=logits.device, dtype=torch.long)
        teacher_logp = torch.as_tensor(support["logprobs"], device=logits.device, dtype=torch.float32).detach()
        shape = (length, args.opd_log_prob_top_k)
        if length == 0:
            ids, teacher_logp = ids.reshape(shape), teacher_logp.reshape(shape)
        if ids.shape != shape or teacher_logp.shape != shape:
            raise ValueError(f"Teacher support must have shape {shape}, got {ids.shape} and {teacher_logp.shape}.")
        student_logp = student_logits.float().log_softmax(dim=-1).gather(1, ids)
        padding = teacher_logp.isneginf()
        probabilities = teacher_logp.exp()
        difference = teacher_logp.masked_fill(padding, 0) - student_logp.masked_fill(padding, 0)
        yield probabilities * difference, probabilities


def forward_kl_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce unnormalized teacher top-k forward KL with the training loss mask."""
    losses, coverage = [], []
    for terms, probabilities in iter_forward_kl_terms(args, batch, logits):
        losses.append(terms.sum(dim=-1))
        coverage.append(probabilities.sum(dim=-1))
    loss = sum_of_sample_mean(torch.cat(losses))
    return loss, {
        "opd_forward_kl": loss.detach(),
        "opd_teacher_coverage": sum_of_sample_mean(torch.cat(coverage)).detach(),
    }
