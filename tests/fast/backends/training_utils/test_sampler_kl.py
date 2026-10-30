"""Independent value/gradient oracles for the three sampler-KL estimators."""

import math
from argparse import ArgumentParser, Namespace

import pytest
import torch

from miles.backends.training_utils.loss.hub.score_centering.sampler_kl import (
    sampled_sampler_kl_losses,
    sampler_kl_loss,
)
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.score_centering import validate_score_centering_args


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("estimator", [0, 1])
def test_sampler_expected_value_and_gradient_match_dense_kl(seed: int, estimator: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 7, generator=generator, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = torch.randn(3, 7, generator=generator).log_softmax(-1).requires_grad_()
    values = sampled_sampler_kl_losses(logp.flatten(), logq.flatten(), torch.ones(21, dtype=torch.bool))
    expected_sample = (logq.detach().exp() * values[estimator].reshape_as(logp)).sum()
    dense = (logq.detach().exp() * (logq.detach() - logp)).sum()
    torch.testing.assert_close(expected_sample, dense)
    actual_grad, sampler_grad = torch.autograd.grad(
        expected_sample, (logits, logq), retain_graph=True, allow_unused=True
    )
    torch.testing.assert_close(actual_grad, torch.autograd.grad(dense, logits)[0])
    assert sampler_grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_uncapped_per_sample_gradients_and_negative_k1(dtype: torch.dtype) -> None:
    logq = torch.tensor([-3.0, -3.0], dtype=dtype, requires_grad=True)
    logp = torch.tensor([-3 + math.log(5), -3 + math.log(0.1)], dtype=dtype, requires_grad=True)
    k1, k3 = sampled_sampler_kl_losses(logp, logq, torch.ones(2, dtype=torch.bool))
    ratio = (logp.float() - logq.detach().float()).exp()
    assert k1.dtype == k3.dtype == torch.float32
    assert k1[0] < 0 and k3[0] > 2  # A cap at two would change this selected objective.
    torch.testing.assert_close(torch.autograd.grad(k1.sum(), logp, retain_graph=True)[0], -torch.ones_like(logp))
    torch.testing.assert_close(torch.autograd.grad(k3.sum(), logp)[0], (ratio - 1).to(dtype))
    assert logq.grad is None


def test_equal_policy_single_sample_k1_gradient_does_not_cancel() -> None:
    logits = torch.tensor([-0.3, 0.7, 0.2], requires_grad=True)
    logp = logits.log_softmax(-1)[:1]
    k1, k3 = sampled_sampler_kl_losses(logp, logp.detach(), torch.tensor([True]))
    assert k1.item() == k3.item() == 0
    k1_gradient = torch.autograd.grad(k1.sum(), logits, retain_graph=True)[0]
    assert k1_gradient.abs().max() > 0.1
    torch.testing.assert_close(torch.autograd.grad(k3.sum(), logits)[0], torch.zeros_like(logits))


def test_masking_happens_before_nonfinite_padding_arithmetic() -> None:
    logp = torch.tensor([-1.0, float("inf"), float("nan"), -float("inf")], requires_grad=True)
    logq = torch.tensor([-1.2, float("inf"), -float("inf"), float("nan")], requires_grad=True)
    k1, k3 = sampled_sampler_kl_losses(logp, logq, torch.tensor([True, False, False, False]))
    assert torch.isfinite(k1).all() and torch.isfinite(k3).all()
    assert (k1[1:] == 0).all() and (k3[1:] == 0).all()
    gradient, sampler_gradient = torch.autograd.grad((k1 + k3).sum(), (logp, logq), allow_unused=True)
    assert torch.isfinite(gradient).all() and (gradient[1:] == 0).all()
    assert sampler_gradient is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_active_token_fails_instead_of_being_dropped(value: float) -> None:
    with pytest.raises(FloatingPointError, match="Non-finite active"):
        sampled_sampler_kl_losses(torch.tensor([value]), torch.tensor([-2.0]), torch.tensor([True]))


def test_fp32_ratio_overflow_is_explicit_and_not_clipped() -> None:
    with pytest.raises(FloatingPointError, match="overflow.*no ratio clipping"):
        sampled_sampler_kl_losses(torch.tensor([-1.0]), torch.tensor([-101.0]), torch.tensor([True]))


def test_near_equal_ratio_uses_stable_expm1() -> None:
    delta = torch.tensor([1e-4, -1e-4, 1e-6, -1e-6])
    _, actual = sampled_sampler_kl_losses(delta, torch.zeros(4), torch.ones(4, dtype=torch.bool))
    expected = (delta.double().expm1() - delta.double()).float()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0.01)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[tokens\]"):
        sampled_sampler_kl_losses(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2))


@pytest.mark.parametrize("size", [0, 3])
def test_no_active_tokens_has_zero_loss_and_gradient(size: int) -> None:
    logp = torch.full((size,), float("nan"), requires_grad=True)
    logq = torch.full((size,), float("inf"), requires_grad=True)
    k1, k3 = sampled_sampler_kl_losses(logp, logq, torch.zeros(size, dtype=torch.bool))
    assert k1.sum() == k3.sum() == 0
    gradient, sampler_gradient = torch.autograd.grad((k1 + k3).sum(), (logp, logq), allow_unused=True)
    torch.testing.assert_close(gradient, torch.zeros_like(logp))
    assert sampler_gradient is None


@pytest.mark.parametrize("partial", [False, True])
def test_head_tail_value_and_fixed_sampler_gradient_match_dense(partial: bool) -> None:
    logits = torch.tensor([[0.4, -0.3, 0.7, -0.8]], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.tensor([[0.2, 0.3, 0.4, 0.1]], dtype=torch.float64)
    ids = [0, 2] if partial else [0, 1, 2, 3]
    if partial:
        # The omitted sampler probabilities follow the documented proportional-tail model.
        q[:, [1, 3]] = 0.4 * logp.detach().exp()[:, [1, 3]] / logp.detach().exp()[:, [1, 3]].sum()
    logq = q.log().requires_grad_()
    actual = sampler_kl_loss(
        logp[:, ids], logq[:, ids], torch.ones(1, len(ids), dtype=torch.bool), torch.tensor([True])
    )
    expected = (q * (q.log() - logp)).sum()
    torch.testing.assert_close(actual.sum(), expected)
    gradient, q_gradient = torch.autograd.grad(actual.sum(), (logits, logq), allow_unused=True, retain_graph=True)
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, logits)[0])
    assert q_gradient is None


def test_head_tail_masks_nonfinite_padding() -> None:
    logits = torch.tensor([[0.4, -0.3], [0.2, 0.1]], requires_grad=True)
    logp = logits.log_softmax(-1)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    padded = torch.cat((logp, torch.full((2, 1), float("nan"))), -1)
    values = sampler_kl_loss(padded, padded.detach(), valid, torch.tensor([True, False]))
    torch.testing.assert_close(values, torch.zeros_like(values))
    torch.testing.assert_close(torch.autograd.grad(values.sum(), logits)[0], torch.zeros_like(logits))


def test_cli_defaults_and_choices() -> None:
    parser = ArgumentParser()
    AlgoConfig.add_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.sampler_kl_coef == 0 and defaults.sampler_kl_estimator == "head_tail"
    for estimator in ["head_tail", "sampled_k1", "sampled_k3"]:
        args = parser.parse_args(["--sampler-kl-coef", ".03", "--sampler-kl-estimator", estimator])
        assert args.sampler_kl_coef == 0.03 and args.sampler_kl_estimator == estimator
    with pytest.raises(SystemExit):
        parser.parse_args(["--sampler-kl-estimator", "unknown"])


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_coefficient_fails_at_startup(coefficient: float) -> None:
    with pytest.raises(ValueError, match="sampler-kl-coef"):
        validate_score_centering_args(Namespace(sampler_kl_coef=coefficient, sampler_kl_estimator="head_tail"))


@pytest.mark.parametrize("coefficient,estimator", [(0.03, "head_tail"), (0, "sampled_k1"), (0, "sampled_k3")])
def test_sampler_kl_is_rejected_for_other_losses(coefficient: float, estimator: str) -> None:
    with pytest.raises(ValueError, match="requires.*score_centering"):
        validate_score_centering_args(
            Namespace(loss_type="policy_loss", sampler_kl_coef=coefficient, sampler_kl_estimator=estimator)
        )
