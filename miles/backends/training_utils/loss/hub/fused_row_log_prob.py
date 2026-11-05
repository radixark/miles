"""`--log-probs-backend fused` on `PaddedRows`: the padded `[T, V]` logits scored by one compiled program per shape.

The fp32 upcast and softmax live only inside each program: the backward saves the bf16 logits and the `[T]`
logsumexp, nothing per-vocabulary. On CUDA the same flag selects the Triton kernel instead (`logit_processors.py`).
"""

import torch

from miles.platforms import current_platform


def _compiled(fn):
    # one program per padded shape where shapes are static; eager elsewhere (the CPU tests)
    if current_platform.dynamic_shapes:
        return fn
    return torch.compile(fn, backend=current_platform.compile_backend, fullgraph=True, dynamic=False)


@_compiled
def _forward(logits: torch.Tensor, targets: torch.Tensor, inv_temperature: torch.Tensor, with_entropy: bool):
    x = logits.float() * inv_temperature
    lse = torch.logsumexp(x, dim=-1)
    log_probs = x.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - lse
    entropy = lse - (torch.exp(x - lse.unsqueeze(-1)) * x).sum(-1) if with_entropy else None
    return log_probs, entropy, lse


@_compiled
def _backward(
    grad: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, lse: torch.Tensor, inv_temperature: torch.Tensor
):
    x = logits.float() * inv_temperature
    one_hot = torch.nn.functional.one_hot(targets, logits.size(-1)).to(torch.float32)
    return ((one_hot - torch.exp(x - lse.unsqueeze(-1))) * (grad * inv_temperature).unsqueeze(-1)).to(logits.dtype)


class FusedRowLogProb(torch.autograd.Function):
    """Log-probs of `targets` `[T]` for `[T, V]` logits, and their entropy `[T]` on request, without gradient."""

    @staticmethod
    def forward(ctx, logits, targets, temperature, with_entropy):
        inv_t = torch.full(
            (), 1.0 / temperature if temperature > 0 else 1.0, dtype=torch.float32, device=logits.device
        )
        log_probs, entropy, lse = _forward(logits, targets, inv_t, with_entropy)
        ctx.save_for_backward(logits, targets, lse, inv_t)
        if with_entropy:
            ctx.mark_non_differentiable(entropy)
        return log_probs, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, targets, lse, inv_t = ctx.saved_tensors
        return _backward(grad_log_probs.contiguous(), logits, targets, lse, inv_t), None, None, None
