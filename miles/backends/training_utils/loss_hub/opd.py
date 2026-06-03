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

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
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
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

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
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls
