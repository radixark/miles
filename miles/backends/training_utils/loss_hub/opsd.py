import torch
import torch.distributed as dist
import torch.nn.functional as F


class _ReplicatedSelectedLogitsAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_values: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
        output = local_values.clone()
        dist.all_reduce(output, group=process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def gather_selected_logits(
    *,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    vocab_size: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"OPSD logits must have shape [response, vocab], got {tuple(logits.shape)}.")
    if token_ids.ndim != 2 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            "OPSD teacher token ids must have shape [response, top_k] aligned with logits, "
            f"got ids={tuple(token_ids.shape)} and logits={tuple(logits.shape)}."
        )
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"OPSD teacher token ids must be integer tensors, got {token_ids.dtype}.")
    if vocab_size <= 0:
        raise ValueError(f"OPSD vocab_size must be positive, got {vocab_size}.")
    if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= vocab_size):
        raise ValueError(f"OPSD teacher token ids must be in [0, {vocab_size}).")

    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    rank = dist.get_rank(process_group) if process_group is not None else 0
    local_vocab_size = logits.shape[-1]
    if local_vocab_size * world_size < vocab_size:
        raise ValueError(
            f"OPSD padded vocabulary has {local_vocab_size * world_size} entries, smaller than vocab_size={vocab_size}."
        )

    vocab_start = rank * local_vocab_size
    local_ids = (token_ids - vocab_start).clamp(min=0, max=local_vocab_size - 1)
    owned = (token_ids >= vocab_start) & (token_ids < vocab_start + local_vocab_size)
    local_values = torch.gather(logits, dim=-1, index=local_ids)
    local_values = local_values * owned.to(dtype=local_values.dtype)

    if world_size == 1:
        return local_values
    return _ReplicatedSelectedLogitsAllReduce.apply(local_values, process_group)


def compute_topk_forward_kl(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    pointwise_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_scores.shape != teacher_scores.shape:
        raise ValueError(
            "OPSD student and teacher scores must have matching [response, top_k] shapes, "
            f"got student={tuple(student_scores.shape)} and teacher={tuple(teacher_scores.shape)}."
        )
    if student_scores.ndim != 2 or student_scores.shape[-1] < 2:
        raise ValueError(f"OPSD scores must have shape [response, top_k>=2], got {tuple(student_scores.shape)}.")
    if pointwise_clip < 0:
        raise ValueError(f"OPSD pointwise KL clip must be non-negative, got {pointwise_clip}.")
    if not torch.isfinite(student_scores).all():
        raise ValueError("OPSD student scores contain non-finite values.")
    if not torch.isfinite(teacher_scores).all():
        raise ValueError("OPSD teacher scores contain non-finite values.")

    teacher_log_probs = F.log_softmax(teacher_scores.detach(), dim=-1)
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    contributions = teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    forward_kl = contributions.sum(dim=-1)

    if pointwise_clip == 0:
        return forward_kl, forward_kl, torch.zeros_like(forward_kl)

    clipped = contributions.clamp(max=pointwise_clip)
    clip_fraction = (contributions > pointwise_clip).to(dtype=contributions.dtype).mean(dim=-1)
    return clipped.sum(dim=-1), forward_kl, clip_fraction
