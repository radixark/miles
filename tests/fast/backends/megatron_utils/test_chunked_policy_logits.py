import pytest
import torch

from miles.backends.training_utils.loss_hub import math_utils


def _compute_log_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    process_group,
) -> torch.Tensor:
    del process_group
    return torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=tokens.unsqueeze(-1),
    )


def _run_log_probs(
    source: torch.Tensor,
    tokens: torch.Tensor,
    *,
    upcast_before_chunking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = source.detach().clone().requires_grad_(True)
    temperature = 0.8
    logits = source.float().div(temperature) if upcast_before_chunking else source
    log_probs, _ = math_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        chunk_size=7,
        temperature=1.0 if upcast_before_chunking else temperature,
    )
    log_probs.sum().backward()
    return log_probs.detach(), source.grad.detach()


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16])
def test_chunked_upcast_preserves_log_probs_and_gradients(
    monkeypatch,
    source_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(math_utils, "compute_log_probs", _compute_log_probs)
    generator = torch.Generator().manual_seed(20260802)
    source = torch.randn(19, 31, dtype=source_dtype, generator=generator)
    tokens = torch.randint(0, 31, (19,), generator=generator)

    baseline = _run_log_probs(
        source,
        tokens,
        upcast_before_chunking=True,
    )
    candidate = _run_log_probs(
        source,
        tokens,
        upcast_before_chunking=False,
    )

    assert torch.equal(baseline[0], candidate[0])
    assert torch.equal(baseline[1], candidate[1])
