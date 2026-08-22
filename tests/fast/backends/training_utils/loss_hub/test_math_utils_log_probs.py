from __future__ import annotations

import torch
from megatron.core.fusions import fused_cross_entropy

from miles.backends.training_utils.loss_hub.math_utils import compute_log_probs


def test_no_grad_log_probs_use_the_training_fused_execution_contract(monkeypatch) -> None:
    """Stored log probs use the training fused contract but return no autograd graph."""
    calls: list[tuple[bool, bool]] = []

    def fake_fused_cross_entropy(
        logits: torch.Tensor,
        _tokens: torch.Tensor,
        _process_group: object,
    ) -> torch.Tensor:
        calls.append((torch.is_grad_enabled(), logits.requires_grad))
        return logits.sum(dim=-1) * 0

    monkeypatch.setattr(
        fused_cross_entropy,
        "fused_vocab_parallel_cross_entropy",
        fake_fused_cross_entropy,
    )
    logits = torch.randn(3, 8)
    tokens = torch.tensor([1, 2, 3])

    with torch.no_grad():
        log_probs = compute_log_probs(logits, tokens, process_group=None)

    assert calls == [(True, True)]
    assert not log_probs.requires_grad


def test_training_log_probs_keep_the_existing_autograd_contract(monkeypatch) -> None:
    """Training log probs retain their input graph and fused execution contract."""
    calls: list[tuple[bool, bool]] = []

    def fake_fused_cross_entropy(
        logits: torch.Tensor,
        _tokens: torch.Tensor,
        _process_group: object,
    ) -> torch.Tensor:
        calls.append((torch.is_grad_enabled(), logits.requires_grad))
        return logits.sum(dim=-1) * 0

    monkeypatch.setattr(
        fused_cross_entropy,
        "fused_vocab_parallel_cross_entropy",
        fake_fused_cross_entropy,
    )
    logits = torch.randn(3, 8, requires_grad=True)
    tokens = torch.tensor([1, 2, 3])

    log_probs = compute_log_probs(logits, tokens, process_group=None)

    assert calls == [(True, True)]
    assert log_probs.requires_grad
    log_probs.sum().backward()
    assert logits.grad is not None
