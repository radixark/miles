import pytest
import torch

from miles.backends.training_utils.loss_hub import math_utils


def test_sft_log_probs_upcast_only_each_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_inputs: list[tuple[torch.Size, torch.dtype]] = []

    def fake_compute_log_probs(
        logits: torch.Tensor,
        tokens: torch.Tensor,
        _tp_group: object,
        *,
        sampling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert sampling_mask is None
        kernel_inputs.append((logits.shape, logits.dtype))
        return torch.zeros((tokens.size(0), 1), dtype=logits.dtype)

    monkeypatch.setattr(math_utils, "compute_log_probs", fake_compute_log_probs)
    logits = torch.zeros((5, 8), dtype=torch.bfloat16)
    tokens = torch.zeros(5, dtype=torch.long)

    math_utils.calculate_log_probs_and_entropy(logits, tokens, None, chunk_size=2)

    assert kernel_inputs == [
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([2, 8]), torch.float32),
        (torch.Size([1, 8]), torch.float32),
    ]
