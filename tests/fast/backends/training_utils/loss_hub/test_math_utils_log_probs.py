from __future__ import annotations

import pytest
import torch
from megatron.core.fusions import fused_cross_entropy

from miles.backends.training_utils.loss_hub.math_utils import compute_log_probs


def _record_fused_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool]]:
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
    return calls


def test_stored_log_probs_keep_the_inference_contract_while_the_switch_is_off(monkeypatch) -> None:
    """Without the switch, stored log probs run the fused kernel exactly as inference always did."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8)
    tokens = torch.tensor([1, 2, 3])

    with torch.no_grad():
        log_probs = compute_log_probs(logits, tokens, process_group=None)

    assert calls == [(False, False)]
    assert not log_probs.requires_grad


def test_stored_log_probs_use_the_training_fused_execution_contract_under_the_switch(monkeypatch) -> None:
    """With the switch on, stored log probs take the training fused contract but return no autograd graph."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8)
    tokens = torch.tensor([1, 2, 3])

    with torch.no_grad():
        log_probs = compute_log_probs(logits, tokens, process_group=None, debug_unified_grad_fused_logprob=True)

    assert calls == [(True, True)]
    assert not log_probs.requires_grad


@pytest.mark.parametrize("switched_on", [False, True])
def test_training_log_probs_keep_the_existing_autograd_contract(monkeypatch, switched_on: bool) -> None:
    """Training log probs retain their input graph and fused execution contract on either side of the switch."""
    calls = _record_fused_calls(monkeypatch)
    logits = torch.randn(3, 8, requires_grad=True)
    tokens = torch.tensor([1, 2, 3])

    log_probs = compute_log_probs(logits, tokens, process_group=None, debug_unified_grad_fused_logprob=switched_on)

    assert calls == [(True, True)]
    assert log_probs.requires_grad
    log_probs.sum().backward()
    assert logits.grad is not None
