"""Distributed selected log-softmax and detached candidate PPO primitives."""

import torch
import torch.distributed as dist


class _SelectedLogSoftmax(torch.autograd.Function):
    """Gather K log-probabilities without materializing K copies of the vocabulary.

    Each TP rank computes the same loss. Backward returns the local vocabulary
    gradient, so replicated output gradients must not be reduced across TP.
    """

    @staticmethod
    def forward(ctx, logits, token_ids, group, rank, size):
        values = logits.float()
        maximum = values.max(dim=-1, keepdim=True).values
        if size > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        shifted = values - maximum
        exp_values = shifted.exp()
        denominator = exp_values.sum(dim=-1, keepdim=True)
        if size > 1:
            dist.all_reduce(denominator, group=group)
        local_ids = token_ids - rank * values.size(-1)
        owned = (local_ids >= 0) & (local_ids < values.size(-1))
        safe_ids = local_ids.clamp(0, values.size(-1) - 1)
        selected = shifted.gather(-1, safe_ids).masked_fill(~owned, 0)
        if size > 1:
            dist.all_reduce(selected, group=group)
        ctx.save_for_backward(exp_values / denominator, safe_ids, owned)
        ctx.input_dtype = logits.dtype
        return selected - denominator.log()

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, safe_ids, owned = ctx.saved_tensors
        gradient = -probabilities * grad_output.sum(dim=-1, keepdim=True)
        gradient.scatter_add_(-1, safe_ids, grad_output.masked_fill(~owned, 0))
        return gradient.to(ctx.input_dtype), None, None, None, None


def selected_log_softmax(logits, token_ids, *, group=None, rank=0, size=1):
    if logits.ndim != 2 or token_ids.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected aligned [response, vocabulary] logits and [response, K] candidate IDs")
    if token_ids.dtype != torch.long:
        raise ValueError("Candidate IDs must be int64")
    if token_ids.numel() and ((token_ids < 0).any() or (token_ids >= logits.size(-1) * size).any()):
        raise ValueError("Candidate ID outside the vocabulary")
    return _SelectedLogSoftmax.apply(logits, token_ids, group, rank, size)


def candidate_policy_loss(
    current, old, teacher, *, refresh: bool, eps_low: float, eps_high: float, dual_clip: float | None = None
):
    """Return per-position loss, fixed-target reverse KL, and candidate clip fraction."""
    if current.shape != old.shape or current.shape != teacher.shape or current.ndim != 2:
        raise ValueError("Candidate log-probability tensors must have the same [response, K] shape")
    old, teacher = old.detach(), teacher.detach()
    reward_student = current.detach() if refresh else old
    advantage = (reward_student.softmax(dim=-1) * (teacher - reward_student)).detach()
    # Match the reference PPO guard: finite tail scores can otherwise overflow exp.
    ratio = (current - old).clamp(-20, 20).exp()
    clipped = ratio.clamp(1 - eps_low, 1 + eps_high)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    if dual_clip is not None:
        if not 1 < dual_clip < float("inf"):
            raise ValueError("Dual-clip PPO requires a bound greater than one")
        objective = torch.where(advantage < 0, torch.maximum(objective, dual_clip * advantage), objective)
    clip_fraction = ((ratio - 1 > eps_high) | (1 - ratio > eps_low)).float().mean(dim=-1)
    return -objective.sum(dim=-1), -advantage.sum(dim=-1), clip_fraction
