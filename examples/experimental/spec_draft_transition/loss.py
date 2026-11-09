"""Exact full-vocabulary CE/L1 with bounded logit memory.

The custom backward recomputes head logits in token chunks. It differentiates
both the student hidden states and head weights; teacher inputs remain frozen.
"""

import math

import torch


def ce_weight(method: str, update: int, updates: int) -> float:
    if method == "ce":
        return 1.0
    if method == "l1":
        return 0.0
    if method != "smooth" or updates < 2 or not 0 <= update < updates:
        raise ValueError((method, update, updates))
    return (1.0 + math.cos(math.pi * update / (updates - 1))) / 2.0


class ProbabilityLoss(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, hidden, weight, teacher_hidden, teacher_weight, temperature, mix, normalizer, chunk):
        ctx.weight_dtype = weight.dtype
        weight = weight.to(hidden.dtype)
        teacher_weight = teacher_weight.to(teacher_hidden.dtype)
        ctx.save_for_backward(hidden, weight, teacher_hidden, teacher_weight)
        ctx.temperature, ctx.mix, ctx.normalizer, ctx.chunk = temperature, mix, normalizer, chunk
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(hidden), chunk):
            logq = (hidden[start:start + chunk] @ weight.T).float().log_softmax(-1)
            p = ((teacher_hidden[start:start + chunk] @ teacher_weight.T).float() / temperature).softmax(-1)
            ce = -(p * logq).sum()
            l1 = (p - logq.exp()).abs().sum()
            total += mix * ce + (1 - mix) * l1
        return total / normalizer

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, output_grad):
        hidden, weight, teacher_hidden, teacher_weight = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, len(hidden), ctx.chunk):
            h = hidden[start:start + ctx.chunk]
            q = (h @ weight.T).float().log_softmax(-1).exp()
            p = ((teacher_hidden[start:start + ctx.chunk] @ teacher_weight.T).float() / ctx.temperature).softmax(-1)
            sign = (q - p).sign()
            grad = ctx.mix * (q - p) + (1 - ctx.mix) * q * (sign - (q * sign).sum(-1, keepdim=True))
            grad = (grad * (output_grad / ctx.normalizer)).to(hidden.dtype)
            grad_hidden[start:start + ctx.chunk] = grad @ weight
            grad_weight.addmm_(grad.T.float(), h.float())
        return grad_hidden, grad_weight.to(ctx.weight_dtype), None, None, None, None, None, None


@torch.no_grad()
def probability_metrics(hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor,
                        teacher_weight: torch.Tensor, temperature: float, chunk: int = 32) -> dict[str, float]:
    totals = dict(ce=0.0, kl=0.0, l1=0.0, entropy=0.0, tokens=float(len(hidden)))
    weight = weight.to(hidden.dtype)
    teacher_weight = teacher_weight.to(teacher_hidden.dtype)
    for start in range(0, len(hidden), chunk):
        logq = (hidden[start:start + chunk] @ weight.T).float().log_softmax(-1)
        logp = ((teacher_hidden[start:start + chunk] @ teacher_weight.T).float() / temperature).log_softmax(-1)
        p = logp.exp()
        totals["ce"] += float(-(p * logq).sum())
        totals["kl"] += float((p * (logp - logq)).sum())
        totals["l1"] += float((p - logq.exp()).abs().sum())
        totals["entropy"] += float(-(p * logp).sum())
    return totals
