from argparse import Namespace

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_responses
from miles.backends.training_utils.loss_hub.math_utils import calculate_opd_gather_with_grad
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def opd_topk_placement(args: Namespace) -> str:
    """Where the in-trainer top-k OPD reverse KL enters the update.

    * ``"loss"`` (default): the KL is a LOSS term
      differentiable through the student's current logits.
    * ``"advantage"``: the KL is subtracted from the per-token advantage. This
      is retained ONLY as the ablation control -- the top-k sum has already
      marginalized over the action, so it is a per-position constant and
      ``E_a[A(r)*grad log pi(a|r)] = A(r)*0``: a (very accurate) baseline that
      cannot teach.

    Only meaningful for the in-trainer top-k path; the sampled-token OPD keeps
    the action inside the coefficient and stays on the advantage side.
    """
    return str(getattr(args, "opd_topk_placement", "loss") or "loss")


def validate_opd_topk_placement(args: Namespace) -> None:
    """Refuse the one OPD wiring that provably cannot teach.

    Advantage-side placement is legitimate ONLY for the sampled-token OPD
    (``--opd-topk-in-trainer 0``), where the coefficient `s(a_t) - t(a_t)` keeps
    the action inside `A` and the expected gradient is ~grad KL. Combined with
    an in-trainer top-k the coefficient no longer depends on the sampled token,
    so `E_a[A(a) * grad log pi(a|a)] = A(a) * 0` -- an exact baseline. This fails
    at launch rather than at analysis time.
    """
    placement = opd_topk_placement(args)
    if placement not in ("loss", "advantage"):
        raise ValueError(f"--opd-topk-placement must be 'loss' or 'advantage', got {placement!r}.")
    if not getattr(args, "use_opd", False):
        return
    if int(getattr(args, "opd_topk_in_trainer", 0) or 0) < 0:
        raise ValueError("--opd-topk-in-trainer must be non-negative.")
    if float(getattr(args, "opd_pointwise_clip", 0.0) or 0.0) < 0:
        raise ValueError("--opd-pointwise-clip must be non-negative (0 disables clipping).")
    if placement == "advantage" and int(getattr(args, "opd_topk_in_trainer", 0) or 0) > 0:
        raise ValueError(
            "--opd-topk-placement=advantage requires the sampled-token OPD "
            "(--opd-topk-in-trainer 0). An in-trainer top-k reverse KL has already "
            "marginalized over the action, so on the advantage side it is a "
            "per-position constant with zero expected policy gradient -- a perfect "
            "baseline that teaches nothing, and one that gets MORE perfect as the "
            "top-k grows. Use --opd-topk-placement=loss to keep the top-k (the "
            "differentiable loss term), or --opd-topk-in-trainer 0 to keep the "
            "advantage side with the sampled-token estimator."
        )


def uses_opd_loss_placement(args: Namespace) -> bool:
    """True when the top-k OPD KL is a loss term rather than an advantage shift."""
    return (
        bool(getattr(args, "use_opd", False))
        and int(getattr(args, "opd_topk_in_trainer", 0) or 0) > 0
        and opd_topk_placement(args) == "loss"
    )


def topk_reverse_kl(
    *,
    student_vals: torch.Tensor,
    teacher_vals: torch.Tensor,
    pointwise_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper-formula in-trainer OPD reverse KL over the student's top-k ids.

    rkl[r] = sum_k exp(s_k) * (s_k - t_k), NOT renormalized within the top-k.
    ``pointwise_clip > 0`` caps each per-(position,k) contribution (a guard
    against a single spike token hogging the KL budget) and reports the clipped
    fraction per position.

    Args:
        student_vals: ``[R, K]`` student full-vocab-normalized logprobs at the
            student's own top-k ids.
        teacher_vals: ``[R, K]`` teacher logprobs gathered at the SAME ids.
        pointwise_clip: 0 disables clipping; otherwise max per-entry
            contribution.

    Returns:
        (rkl ``[R]``, clipfrac ``[R]``).
    """
    if student_vals.shape != teacher_vals.shape or student_vals.ndim != 2:
        raise ValueError(
            f"OPD top-k scores must share a [R, K] shape, got student={tuple(student_vals.shape)}"
            f" teacher={tuple(teacher_vals.shape)}"
        )
    if pointwise_clip < 0:
        raise ValueError(f"pointwise_clip must be >= 0, got {pointwise_clip}")
    if not torch.isfinite(student_vals).all():
        raise ValueError("OPD student top-k scores contain non-finite values")
    if not torch.isfinite(teacher_vals).all():
        raise ValueError("OPD teacher top-k scores contain non-finite values")

    contributions = student_vals.exp() * (student_vals - teacher_vals)
    if pointwise_clip == 0:
        return contributions.sum(dim=-1), torch.zeros(
            student_vals.shape[0], dtype=contributions.dtype, device=contributions.device
        )
    clipped = contributions.clamp(max=pointwise_clip)
    clipfrac = (contributions > pointwise_clip).to(dtype=contributions.dtype).mean(dim=-1)
    return clipped.sum(dim=-1), clipfrac


def topk_overlap(
    *,
    student_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Per-position overlap fraction of the teacher's and student's top-k id sets.

    ``overlap[r] = |student_ids[r] ∩ teacher_ids[r]| / K`` -- the distillation health
    metric: distillation transfers through the student's top-k tokens, so for the
    KL to teach anything the teacher's own top-k must increasingly coincide with
    the student's over training -- a flat or falling curve means the knowledge
    view is shifting the teacher's distribution away from the student's output
    habits instead of sharpening it in place.

    Args:
        student_ids: ``[R, K]`` long, the student's top-k ids per position.
        teacher_ids: ``[R, K]`` long, the teacher's top-k ids per position
            (same K; each row is a set -- top-k ids are unique by construction).
        chunk_size: rows compared per broadcast chunk (the ``[r, K, K]``
            comparison cube is memory-bound at K=128).

    Returns:
        ``[R]`` float32 overlap fractions in [0, 1].
    """
    if student_ids.shape != teacher_ids.shape or student_ids.ndim != 2:
        raise ValueError(
            f"top-k id sets must share a [R, K] shape, got student={tuple(student_ids.shape)}"
            f" teacher={tuple(teacher_ids.shape)}"
        )
    num_positions, k = student_ids.shape
    if num_positions == 0:
        return student_ids.new_zeros((0,), dtype=torch.float32)
    parts = []
    with torch.no_grad():
        for s_chunk, t_chunk in zip(
            student_ids.split(chunk_size, dim=0), teacher_ids.split(chunk_size, dim=0), strict=True
        ):
            shared = (s_chunk.unsqueeze(-1) == t_chunk.unsqueeze(-2)).any(dim=-1)
            parts.append(shared.sum(dim=-1).to(torch.float32) / k)
    return torch.cat(parts, dim=0)


def compute_opd_topk_distill(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-sample ``[R]`` top-k reverse KL, differentiable through ``logits``.

    This is the consolidation term of the top-k distillation: the ids and the teacher's logprobs at
    those ids are constants precomputed once per collected batch (the student
    pre-pass and teacher gather), and the student side is re-read from THIS
    training forward so the gradient flows:

        rkl[r] = sum_k exp(s_k) * (s_k - t_k),   s_k from the live logits

    Reading it against ``apply_opd_kl_to_advantages``: identical formula, but
    there the whole thing is a detached number multiplied onto
    ``grad log pi(sampled token)``, which averages to zero over the action. Here
    the derivative is taken of the KL itself, so it is non-zero every step.

    Args:
        args: Needs ``qkv_format``, ``rollout_temperature`` and optionally
            ``opd_pointwise_clip``/``vocab_size`` (``get_responses`` reads the rest).
        batch: The micro-batch; needs ``unconcat_tokens``, ``total_lengths``,
            ``response_lengths``, ``opd_topk_ids`` and ``teacher_opd_gathered_vals``.
        logits: ``[1, T, V_local]`` policy logits from the training forward.

    Returns:
        One ``[R]`` tensor per sample, response-aligned and carrying grad.
    """
    topk_ids = batch.get("opd_topk_ids")
    teacher_vals = batch.get("teacher_opd_gathered_vals")
    if topk_ids is None:
        raise ValueError(
            "OPD loss placement requires opd_topk_ids in the training micro-batch; "
            "the student top-k pre-pass did not run or the key was not requested."
        )
    if teacher_vals is None:
        raise ValueError(
            "OPD loss placement requires teacher_opd_gathered_vals in the training "
            "micro-batch; the teacher gather pass did not run or the key was not requested."
        )

    tp_group = get_parallel_state().tp.group
    pointwise_clip = float(getattr(args, "opd_pointwise_clip", 0.0) or 0.0)
    vocab_size = getattr(args, "vocab_size", None)

    rkls = []
    for sample_idx, (logits_chunk, _tokens_chunk) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
    ):
        student_vals = calculate_opd_gather_with_grad(
            logits_chunk,
            tp_group,
            gather_ids=topk_ids[sample_idx].to(device=logits_chunk.device),
            vocab_size=vocab_size,
            # Same row-chunking knob the no_grad pre-pass uses; <= 0 falls back
            # to the routine's own bounded default (never "unchunked").
            chunk_size=int(getattr(args, "log_probs_chunk_size", -1) or -1),
        )
        rkl, _clipfrac = topk_reverse_kl(
            student_vals=student_vals,
            teacher_vals=teacher_vals[sample_idx].to(device=student_vals.device, dtype=student_vals.dtype),
            pointwise_clip=pointwise_clip,
        )
        rkls.append(rkl)
    return rkls


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

    if student_log_probs is None:
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
