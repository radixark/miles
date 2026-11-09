"""Full-vocabulary CE/L1 using ordinary PyTorch autograd and AMP."""

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


def probability_loss(hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor,
                     teacher_weight: torch.Tensor, temperature: float, mix: float,
                     normalizer: int, chunk: int) -> torch.Tensor:
    """Differentiate student probabilities through the actual autocast forward."""
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(hidden), chunk):
        logits = (hidden[start:start + chunk] @ weight.T).float()
        logq = logits.log_softmax(-1)
        q = logits.softmax(-1)
        with torch.no_grad():
            p = ((teacher_hidden[start:start + chunk] @ teacher_weight.T).float() / temperature).softmax(-1)
        ce = -(p * logq).sum()
        l1 = (p - q).abs().sum()
        total = total + mix * ce + (1 - mix) * l1
    return total / normalizer


@torch.no_grad()
def probability_metrics(hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor,
                        teacher_weight: torch.Tensor, temperature: float, chunk: int = 32) -> dict[str, float]:
    totals = dict(ce=0.0, kl=0.0, l1=0.0, entropy=0.0, tokens=float(len(hidden)))
    weight = weight.to(hidden.dtype)
    teacher_weight = teacher_weight.to(teacher_hidden.dtype)
    for start in range(0, len(hidden), chunk):
        logits = (hidden[start:start + chunk] @ weight.T).float()
        teacher_logits = (teacher_hidden[start:start + chunk] @ teacher_weight.T).float() / temperature
        logq, q = logits.log_softmax(-1), logits.softmax(-1)
        logp, p = teacher_logits.log_softmax(-1), teacher_logits.softmax(-1)
        totals["ce"] += float(-(p * logq).sum())
        totals["kl"] += float((p * (logp - logq)).sum())
        totals["l1"] += float((p - q).abs().sum())
        totals["entropy"] += float(-(p * logp).sum())
    return totals
