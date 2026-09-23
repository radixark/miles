"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import math
from typing import Any

import torch
import torch.distributed as dist


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> torch.Tensor:
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    if mode == "none":
        return torch.ones_like(log_ratio)
    if mode == "tis":
        return log_ratio.clamp(max=math.log(tis_clip)).exp()
    if mode == "mis":
        inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
        return torch.where(inside, log_ratio.clamp(max=math.log(mis_high)).exp(), 0.0)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def score_centering_loss(
    train_log_probs: torch.Tensor,
    train_head_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    mode: str = "none",
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return unreduced token losses and detached token metrics.

    Inputs have shape [tokens], except head tensors/mask [tokens, candidates].
    Missing head slots have a false mask. Tail masses use the appendix's floor;
    cancellation is exact for a full distribution, approximate for a true tail
    that differs from the modeled, rescaled trainer tail.
    """
    head_log_probs = torch.where(head_mask, train_head_log_probs, 0.0)
    with torch.no_grad():
        p = head_log_probs.exp().masked_fill(~head_mask, 0.0)
        q = rollout_head_log_probs.exp().masked_fill(~head_mask, 0.0)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        if mode == "none":
            weighted_q, alpha = q, rho
        elif mode == "tis":
            # q * min(p/q, c), including q=0, without 0 * inf.
            weighted_q, alpha = torch.minimum(p, tis_clip * q), (tis_clip * rho).clamp_max(1)
        elif mode == "mis":
            log_ratio = head_log_probs - rollout_head_log_probs
            inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
            weighted_q = torch.where(inside & head_mask, p, 0.0)
            alpha = ((rho >= 1 / mis_high) & (rho <= 1 / mis_low)).to(p.dtype)
        else:
            raise ValueError(f"Unknown score-centering importance weighting: {mode}")
        residual = weighted_q - alpha.unsqueeze(-1) * p
        weight = importance_weights(
            train_log_probs - rollout_log_probs,
            mode,
            tis_clip=tis_clip,
            mis_low=mis_low,
            mis_high=mis_high,
        )
    correction = (residual * head_log_probs).sum(-1)
    loss = -advantages.detach() * (weight * train_log_probs - correction)
    return loss, {
        "sc_correction": correction.detach(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        "sc_importance_weight": weight,
    }


class _SelectedLogProbs(torch.autograd.Function):
    """Vocabulary-sharded log-softmax gather with a replicated loss on TP ranks.

    Only selected logits and normalization scalars are communicated. Backward
    computes each rank's vocabulary slice directly; reducing identical output
    gradients across TP ranks would incorrectly multiply the gradient by TP.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        vocab_size: int,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.shape[-1]
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        padded = torch.arange(width, device=logits.device) + rank * width >= vocab_size
        work = logits.masked_fill(padded, -torch.inf)
        maximum = work.amax(-1, keepdim=True)
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        probabilities = (work - maximum).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        selected = torch.where(local, work.gather(-1, local_ids), 0.0)
        if group is not None:
            dist.all_reduce(denominator, group=group)
            dist.all_reduce(selected, group=group)
        probabilities.div_(denominator)
        entropy = probabilities.new_zeros(probabilities.size(0))
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, probabilities.log(), 0.0)).sum(-1)
            if group is not None:
                dist.all_reduce(entropy, group=group)
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid, entropy)
        return torch.where(valid, selected - maximum - denominator.log(), 0.0), entropy

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        grad_logits = -probabilities * grad.sum(-1, keepdim=True)
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            grad_logits -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        return grad_logits, None, None, None, None


def selected_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
) -> torch.Tensor:
    """Log-probabilities at global IDs [T, K] from local logits [T, V/TP].

    -1 is a padding ID and produces zero with zero gradient. The true vocabulary
    size excludes Megatron's padded vocabulary entries from normalization.
    """
    return selected_log_probs_and_entropy(
        logits, token_ids, group=group, vocab_size=vocab_size, temperature=temperature, chunk_size=chunk_size
    )[0]


def selected_log_probs_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
    with_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected logprobs and optional entropy of the same unpadded distribution."""
    size = dist.get_world_size(group) if group is not None else 1
    vocab_size = vocab_size or logits.size(-1) * size
    if temperature <= 0 or not 0 < vocab_size <= logits.size(-1) * size:
        raise ValueError("Score centering needs a positive temperature and valid vocabulary size")
    if token_ids.ndim != 2 or logits.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected logits [T, V/TP] and selected token IDs [T, K]")
    if ((token_ids < -1) | (token_ids >= vocab_size)).any():
        raise ValueError("Score-centering token ID is outside the model vocabulary")
    if logits.size(0) == 0:
        return logits.sum(-1, keepdim=True).expand_as(token_ids), logits.sum(-1)
    chunk_size = chunk_size if chunk_size > 0 else logits.size(0)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    chunks = [
        _SelectedLogProbs.apply(chunk.to(dtype) / temperature, ids, group, vocab_size, with_entropy)
        for chunk, ids in zip(logits.split(chunk_size), token_ids.split(chunk_size), strict=True)
    ]
    return torch.cat([chunk[0] for chunk in chunks]), torch.cat([chunk[1] for chunk in chunks])
