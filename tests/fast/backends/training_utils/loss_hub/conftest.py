from collections.abc import Callable

import pytest
import torch
from megatron.core.fusions import fused_cross_entropy


@pytest.fixture
def fused_cross_entropy_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[bool, bool]]:
    calls: list[tuple[bool, bool]] = []

    def fake_fused_cross_entropy(
        logits: torch.Tensor,
        tokens: torch.Tensor,
        process_group: object,
    ) -> torch.Tensor:
        del tokens, process_group
        calls.append((torch.is_grad_enabled(), logits.requires_grad))
        return logits.sum(dim=-1) * 0

    fake: Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor] = fake_fused_cross_entropy
    monkeypatch.setattr(fused_cross_entropy, "fused_vocab_parallel_cross_entropy", fake)
    return calls
