"""Compare the implemented gradient with independent dense-distribution oracles."""

import pytest
import torch

from miles.backends.training_utils.loss_hub.score_centering import (
    score_centering_loss,
    selected_log_probs,
    selected_log_probs_and_entropy,
)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_gradient_matches_reconstructed_full_distribution(mode: str, k: int, device: str) -> None:
    generator = torch.Generator().manual_seed(27)
    logits = torch.randn(4, 7, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    q = torch.softmax(torch.randn(4, 7, generator=generator, dtype=torch.float64).to(device) * 2, -1)
    q_head, ids = q.topk(k, dim=-1)
    sampled = q.argmin(-1)  # outside the retained head when k < vocabulary size
    advantage = torch.tensor([1.7, -0.8, 0.0, 0.2], dtype=torch.float64, device=device)
    logp = logits.log_softmax(-1)
    loss, _ = score_centering_loss(
        logp.gather(-1, sampled[:, None]).squeeze(-1),
        logp.gather(-1, ids),
        q.gather(-1, sampled[:, None]).squeeze(-1).log(),
        q_head.log(),
        torch.ones_like(ids, dtype=torch.bool),
        advantage,
        mode=mode,
    )
    actual_grad = torch.autograd.grad(loss.sum(), logits)[0]

    reference_logits = logits.detach().clone().requires_grad_()
    reference_logp = reference_logits.log_softmax(-1)
    p = reference_logp.detach().exp()
    rho = (1 - q_head.sum(-1)).clamp_min(1e-6) / (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
    q_hat = p * rho[:, None]
    q_hat.scatter_(-1, ids, q_head)
    ratio = p / q_hat
    sampled_ratio = p.gather(-1, sampled[:, None]).squeeze(-1) / q.gather(-1, sampled[:, None]).squeeze(-1)
    if mode == "none":
        weights, sample_weights = torch.ones_like(ratio), torch.ones_like(sampled_ratio)
    elif mode == "tis":
        weights, sample_weights = ratio.clamp_max(2), sampled_ratio.clamp_max(2)
    else:
        weights = torch.where((ratio >= 0.5) & (ratio <= 5), ratio, 0.0)
        sample_weights = torch.where((sampled_ratio >= 0.5) & (sampled_ratio <= 5), sampled_ratio, 0.0)
    reference = -advantage * (
        sample_weights * reference_logp.gather(-1, sampled[:, None]).squeeze(-1)
        - (q_hat * weights * reference_logp).sum(-1)
    )
    expected_grad = torch.autograd.grad(reference.sum(), reference_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_full_distribution_has_zero_constant_reward_gradient(mode: str) -> None:
    logits = torch.tensor([0.7, -1.2, 2.3, 0.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.1, 0.5, 0.15, 0.25], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    loss, _ = score_centering_loss(
        logp,
        logp.expand(4, 4),
        q.log(),
        q.log().expand(4, 4),
        torch.ones(4, 4, dtype=torch.bool),
        torch.ones(4),
        mode=mode,
    )
    gradient = torch.autograd.grad((q * loss).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_matched_policies_have_no_correction() -> None:
    logp = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64).log().requires_grad_()
    loss, metrics = score_centering_loss(
        logp[:, 2],
        logp[:, :2],
        logp[:, 2].detach(),
        logp[:, :2].detach(),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(metrics["sc_correction"], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(loss, -2 * logp[:, 2])


def test_unclipped_importance_sampling_has_no_correction() -> None:
    p = torch.tensor([[0.55, 0.3, 0.15]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    _, metrics = score_centering_loss(
        p[:, 2].log(),
        p[:, :2].log(),
        q[:, 2].log(),
        q[:, :2].log(),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1),
        mode="tis",
        tis_clip=10,
    )
    torch.testing.assert_close(metrics["sc_correction"], torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_tiny_tails_and_padding_are_finite_and_detached(mode: str) -> None:
    logits = torch.tensor([[40.0, -40.0, -80.0]], requires_grad=True)
    head = logits.log_softmax(-1)[:, :2]
    q = torch.tensor([[0.0, -torch.inf]], requires_grad=True)
    sampled_q = torch.tensor([-0.1], requires_grad=True)
    advantage = torch.ones(1, requires_grad=True)
    loss, metrics = score_centering_loss(
        head[:, 0],
        head,
        sampled_q,
        q,
        torch.tensor([[True, False]]),
        advantage,
        mode=mode,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert q.grad is None and sampled_q.grad is None and advantage.grad is None


@pytest.mark.parametrize("chunk_size", [-1, 1, 3])
def test_selected_log_probs_and_gradients_match_dense(chunk_size: int) -> None:
    logits = torch.randn(5, 9, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 3, -1, 0], [1, 2, 6, 2], [3, 3, 3, 3], [6, -1, 1, 2], [5, 2, 4, 1]])
    weights = torch.randn(5, 4, dtype=torch.float64)
    actual = selected_log_probs(logits, ids, vocab_size=7, temperature=0.7, chunk_size=chunk_size)
    reference = (logits[:, :7] / 0.7).log_softmax(-1).gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    torch.testing.assert_close(actual, reference)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((reference * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (actual_grad[:, 7:] == 0).all()


def test_entropy_uses_same_unpadded_distribution_and_gradient() -> None:
    logits = torch.randn(3, 9, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2], [1, 4], [3, 6]])
    selected, entropy = selected_log_probs_and_entropy(logits, ids, vocab_size=7, temperature=0.7, with_entropy=True)
    logp = (logits[:, :7] / 0.7).log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(entropy, expected_entropy)
    actual_grad = torch.autograd.grad(selected.sum() - 0.2 * entropy.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(logp.gather(-1, ids).sum() - 0.2 * expected_entropy.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_selected_log_probs_gradcheck_and_empty_response() -> None:
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2, -1], [1, 3, 1]])
    assert torch.autograd.gradcheck(lambda x: selected_log_probs(x, ids), (logits,))
    empty = logits[:0]
    result = selected_log_probs(empty, ids[:0])
    assert result.shape == (0, 3)
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
