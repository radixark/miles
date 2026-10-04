"""Support scoring against an independent dense masked-softmax oracle."""

import pytest
import torch

from miles.backends.training_utils.loss.hub.score_centering import selected_log_probs_and_entropy


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
@pytest.mark.parametrize("chunk_size", [1, -1])
@pytest.mark.parametrize("with_entropy", [False, True])
def test_support_scores_gradients_and_saved_state(dtype, temperature, chunk_size, with_entropy):
    logits = torch.randn(4, 31, generator=torch.Generator().manual_seed(72), dtype=dtype).requires_grad_()
    ids = torch.tensor([[3, 3, 7, 11], [8, 8, -1, -1], [2, -1, -1, -1], [7, 7, 3, -1]])
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value.shape) or value, lambda value: value
    ):
        actual, entropy = selected_log_probs_and_entropy(
            logits,
            ids,
            vocab_size=29,
            temperature=temperature,
            chunk_size=chunk_size,
            with_entropy=with_entropy,
            sampling_support=True,
        )
    assert all(31 not in shape for shape in saved)
    reference_logits = logits.detach().clone().requires_grad_()
    support = torch.zeros_like(logits, dtype=torch.bool)
    for row in range(len(ids)):
        support[row, ids[row, 1:][ids[row, 1:] >= 0]] = True
    active = support.any(-1)
    dtype_work = torch.float64 if dtype == torch.float64 else torch.float32
    work = (reference_logits.to(dtype_work) / temperature).masked_fill(~support, -torch.inf)
    work = torch.where(active[:, None], work, 0.0)
    logp = work.log_softmax(-1)
    reference = logp.gather(-1, ids.clamp_min(0)).masked_fill((ids < 0) | ~active[:, None], 0)
    safe_logp = logp.masked_fill(~support, 0)
    expected_entropy = -(logp.exp() * safe_logp).sum(-1) if with_entropy else entropy.new_zeros(4)
    weights = torch.randn(actual.shape, generator=torch.Generator().manual_seed(4), dtype=dtype_work)
    ((actual * weights).sum() - 0.2 * entropy.sum()).backward()
    ((reference * weights).sum() - 0.2 * expected_entropy.sum()).backward()
    tolerance = 0.02 if dtype == torch.bfloat16 else (1e-12 if dtype == torch.float64 else 2e-6)
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(entropy, expected_entropy, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(logits.grad[~support]) == 0


def test_empty_support_response():
    logits = torch.empty(0, 31, requires_grad=True)
    selected, entropy = selected_log_probs_and_entropy(
        logits, torch.empty(0, 4, dtype=torch.long), sampling_support=True, with_entropy=True
    )
    (selected.sum() + entropy.sum()).backward()
    assert logits.grad.shape == logits.shape
