"""Regression coverage for reference KL and finite score-centering gradients."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub import score_centering_loss as loss_module
from miles.backends.training_utils.loss.hub.score_centering_loss import _regularization


def test_candidate_collection_retains_full_vocabulary_kl_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[0.2, -0.3, 0.8, 0.4, 10.0]], requires_grad=True)
    args = Namespace(
        use_kl_loss=True,
        entropy_coef=0,
        observe_training_entropy=False,
        use_sampling_support_replay=True,
        vocab_size=4,
        rollout_temperature=0.7,
        log_probs_chunk_size=2,
        allgather_cp=False,
    )
    parallel = SimpleNamespace(tp=SimpleNamespace(size=1, group=None), cp=SimpleNamespace(size=1))
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(
        loss_module,
        "_iter_response_chunks",
        lambda *unused, **kwargs: iter([(logits, torch.tensor([2]), [0])]),
    )
    batch = {
        "rollout_topk_token_ids": [torch.tensor([[2, 3, -1]])],
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
    }
    probabilities = loss_module._candidate_log_probs(args, batch, logits)
    full = (logits[:, :4] / 0.7).log_softmax(-1)[:, 2]
    support = (logits[:, [2, 3]] / 0.7).log_softmax(-1)[:, 0]
    torch.testing.assert_close(probabilities["kl_log_probs"][0], full)
    torch.testing.assert_close(probabilities["selected"][0][:, 0], support)
    actual = torch.autograd.grad(probabilities["kl_log_probs"][0].sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.autograd.grad(full.sum(), logits)[0])


@pytest.mark.parametrize("kind", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("unbiased", [False, True])
def test_reference_kl_matches_full_vocabulary_gradient(kind: str, unbiased: bool) -> None:
    logits = torch.tensor([0.2, -0.3, 0.8], requires_grad=True)
    full = logits.log_softmax(-1)[0:1]
    support = logits[:2].log_softmax(-1)[0:1]
    rollout = torch.tensor([-0.5])
    reference = torch.tensor([-1.3])
    args = Namespace(use_kl_loss=True, use_unbiased_kl=unbiased, kl_loss_type=kind, kl_loss_coef=0.1)
    loss, _ = _regularization(
        args, {"ref_log_probs": [reference]}, None, support, rollout, torch.tensor([True]), torch.mean, full
    )
    difference = full - reference
    if kind == "k1":
        expected = difference
    elif kind == "k2":
        expected = difference.square() / 2
    else:
        expected = (-difference).exp() - 1 + difference
    if unbiased:
        expected = expected * (support - rollout).exp()
    if kind == "low_var_kl":
        expected = expected.clamp(-10, 10)
    expected = 0.1 * expected.mean()
    torch.testing.assert_close(loss, expected)
    actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("coefficient", [0.0, 0.1])
@pytest.mark.parametrize("unbiased", [False, True])
def test_extreme_reference_kl_has_finite_loss_and_gradients(coefficient: float, unbiased: bool) -> None:
    log_probs = torch.tensor([-100.0, -1.0, -torch.inf, -0.5], requires_grad=True)
    reference = torch.tensor([0.0, -0.8, -torch.inf, -torch.inf])
    rollout = torch.tensor([-200.0, -1.1, -torch.inf, -torch.inf])
    args = Namespace(use_kl_loss=True, use_unbiased_kl=unbiased, kl_loss_type="k3", kl_loss_coef=coefficient)
    loss, metrics = _regularization(
        args,
        {"ref_log_probs": [reference]},
        None,
        log_probs,
        rollout,
        torch.tensor([True, True, False, True]),
        torch.mean,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])
    if coefficient:
        loss.backward()
        assert torch.isfinite(log_probs.grad).all()
        assert log_probs.grad[0] == log_probs.grad[2] == log_probs.grad[3] == 0
        assert log_probs.grad[1] != 0
    else:
        assert loss == 0


def test_identical_full_vocabulary_policies_have_zero_kl_under_replay() -> None:
    logits = torch.tensor([0.2, -0.3, 0.8], requires_grad=True)
    full = logits.log_softmax(-1)[0:1]
    support = logits[:2].log_softmax(-1)[0:1]
    args = Namespace(use_kl_loss=True, use_unbiased_kl=False, kl_loss_type="k2", kl_loss_coef=0.1)
    loss, metrics = _regularization(
        args,
        {"ref_log_probs": [full.detach()]},
        None,
        support,
        support.detach(),
        torch.tensor([True]),
        torch.mean,
        full,
    )
    assert metrics["kl_loss"] == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
