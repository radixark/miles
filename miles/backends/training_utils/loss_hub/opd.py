from argparse import Namespace

import torch

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

    ``--opd-kl-clip`` bounds each per-token contribution before it reaches the
    advantage. Token-level divergence is heavy-tailed, and unbounded the update is
    carried by a handful of stylistic tokens rather than by the content ones. The
    top-k path clips per vocabulary entry while building ``opd_reverse_kl``, so the
    ceiling is applied there and this function leaves those values alone.

    Args:
        args: Configuration containing `use_opd`, `opd_kl_coef` and `opd_kl_clip`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of old-student log-probability tensors. OPD
            treats these as fixed scoring inputs.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    if getattr(args, "opd_divergence", "reverse_kl") == "forward_kl":
        # Forward KL is minimised directly in the loss, where the student's logits are
        # still differentiable. It never produces teacher_log_probs, so there is nothing
        # to fold into the advantages here.
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

    clip = getattr(args, "opd_kl_clip", None)
    reverse_kls = []
    clip_flags = []
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
        if clip is not None:
            # Logged so the ceiling is visible: a clipfrac near 0 means tau is inert and
            # near 1 means it is flattening the signal instead of just its tail.
            clip_flags.append((reverse_kl > clip).to(reverse_kl.dtype))
            reverse_kl = reverse_kl.clamp(max=clip)
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging.
    rollout_data["opd_reverse_kl"] = reverse_kls
    if clip_flags:
        rollout_data["opd_kl_clipfrac"] = clip_flags


def forward_kl_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(teacher || student) over the teacher's top-k support, per response token.

    The teacher supplies ``p_T`` on its own top-k; the student's log-probs at those same
    ids come from the training logits, so this is differentiable in the student and needs
    no second scoring call. Returns ``(loss, clipfrac, teacher_coverage)``.

    Forward KL is mode-covering: it penalises the student for putting too little mass
    where the teacher puts some, which is the opposite of the sampled-token reverse-KL
    signal and the direction the OPSD paper adopts.
    """
    top_ids = batch.get("teacher_top_ids")
    if top_ids is None:
        raise ValueError(
            "--opd-divergence=forward_kl is set but the batch carries no teacher_top_ids. "
            "The teacher support is dropped unless the field is listed in the get_batch key "
            "set; silently skipping it would train on a zero loss."
        )
    top_logprobs = batch["teacher_top_logprobs"]

    # Local import: loss_hub.opd is imported by loss.py before logit_processors is needed.
    from miles.backends.training_utils.loss_hub.logit_processors import get_responses

    clip = getattr(args, "opd_kl_clip", None)
    per_token, clipped, covered = [], [], []
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    for i, (logits_chunk, _) in enumerate(responses):
        ids, t_logp = top_ids[i], top_logprobs[i]
        if ids.numel() == 0:
            per_token.append(logits_chunk.new_zeros((0,)))
            clipped.append(logits_chunk.new_zeros((0,)))
            covered.append(logits_chunk.new_zeros((0,)))
            continue
        # Validated to be unsharded, so a plain log_softmax is the true distribution.
        student_logp = torch.log_softmax(logits_chunk.float(), dim=-1).gather(1, ids)
        # Padding carries -inf. Mask it rather than letting arithmetic handle it: p_t would
        # be 0 and the difference -inf, and 0 * -inf is NaN. Masking also keeps a genuine
        # +inf (student assigning ~zero mass where the teacher has some) intact, which
        # nan_to_num would have silently zeroed -- that is the largest signal there is, and
        # the clip below is what bounds it.
        valid = torch.isfinite(t_logp)
        p_t = torch.where(valid, t_logp.exp(), torch.zeros_like(t_logp))
        contribution = torch.where(valid, p_t * (t_logp - student_logp), torch.zeros_like(student_logp))
        if clip is not None:
            clipped.append((contribution > clip).to(contribution.dtype).mean(dim=-1))
            contribution = contribution.clamp(max=clip)
        per_token.append(contribution.sum(dim=-1))
        # Teacher mass inside the support. The reference implementation sums over the full
        # vocabulary; this truncated sum stands in for it only while coverage is near 1.0,
        # so it is measured rather than assumed. Deliberately NOT renormalised: their
        # tau is calibrated to full-vocabulary p_T magnitudes, and renormalising over a
        # small support inflates every contribution so the same tau over-clips.
        covered.append(p_t.sum(dim=-1))

    loss = sum_of_sample_mean(torch.cat(per_token, dim=0))
    clipfrac = sum_of_sample_mean(torch.cat(clipped, dim=0)) if clipped else loss.new_zeros(())
    coverage = sum_of_sample_mean(torch.cat(covered, dim=0)) if covered else loss.new_zeros(())
    return loss, clipfrac, coverage
