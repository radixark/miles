"""Check BPO against the public loss and dense policy gradients."""

import argparse
import math

import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import losses as loss_utils
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("bpo") / "rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    singleton = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            **{name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "pp", "ep", "etp", "indep_dp")},
            tp=GroupInfo(rank=0, size=1, group=dist.group.WORLD),
        ),
    )


def _token_loss(
    monkeypatch,
    scores,
    rollout_scores,
    advantages,
    smoothing,
    cap,
    low=0.2,
    high=0.28,
    use_rollout_logprobs=True,
    trainer_scores=None,
):
    args = make_args(
        loss_type="bpo",
        use_rollout_logprobs=use_rollout_logprobs,
        eps_clip=low,
        eps_clip_high=high,
        bpo_smoothing=smoothing,
        bpo_weight_cap=cap,
        entropy_coef=0.0,
        observe_training_entropy=False,
    )
    n = scores.numel()
    batch = dict(
        advantages=[advantages],
        rollout_log_probs=[rollout_scores] if rollout_scores is not None else None,
        log_probs=[trainer_scores] if trainer_scores is not None else None,
        unconcat_tokens=[torch.zeros(n + 1, dtype=torch.long)],
        total_lengths=[n + 1],
        response_lengths=[n],
        loss_masks=[torch.ones(n)],
    )
    monkeypatch.setattr(loss_utils, "get_log_probs_and_entropy", lambda *a, **kw: {"log_probs": [scores]})
    monkeypatch.setattr(loss_utils, "compute_ess_ratio_contribution", lambda **kw: scores.new_tensor(1.0))
    return get_loss_function(args)(args, batch, torch.zeros(1, n + 1, 2), torch.sum)


@pytest.mark.parametrize("smoothing,cap", [(0.1, 3.0), (0.05, 2.0)])
def test_bpo_matches_scalar_formula_and_stops_weight_gradients(monkeypatch, smoothing, cap):
    probabilities = [0.1, 0.5, 0.8, 0.95, 0.999999, 1e-30]
    rollout = [0.9, 0.4, 0.1, 0.1, 0.01, 1e-20]
    p = torch.tensor(probabilities * 3, dtype=torch.float64).log().requires_grad_()
    q = torch.tensor(rollout * 3, dtype=torch.float64).log().requires_grad_()
    advantages = torch.tensor([1.0] * 6 + [-2.0] * 6 + [0.0] * 6, dtype=torch.float64, requires_grad=True)
    loss, metrics = _token_loss(monkeypatch, p, q, advantages, smoothing, cap)
    coefficients, expected_clipped = [], []
    for pi, mu, advantage in zip(probabilities * 3, rollout * 3, advantages.tolist(), strict=True):
        weight = (1 + smoothing - mu) / (1 + smoothing - pi)
        masked = (advantage > 0 and weight > 1.28) or (advantage < 0 and weight < 0.8)
        coefficients.append(0.0 if masked else -advantage * min(weight, cap))
        expected_clipped.append(float(masked))
    expected = torch.tensor(coefficients, dtype=torch.float64)
    torch.testing.assert_close(loss, (expected * p.detach()).sum())
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(expected_clipped, dtype=torch.float64).sum())
    gradient, q_gradient, a_gradient = torch.autograd.grad(loss.sum(), (p, q, advantages), allow_unused=True)
    torch.testing.assert_close(gradient, expected)
    assert q_gradient is None and a_gradient is None


def test_bpo_mask_boundaries_and_zero_advantages(monkeypatch):
    p = torch.tensor([0.0, -torch.inf, math.log(0.5), math.log(0.5)], requires_grad=True)
    q = torch.tensor([0.0, -torch.inf, math.log(0.5), math.log(0.5)])
    advantages = torch.tensor([0.0, 0.0, 1.0, -1.0])
    loss, metrics = _token_loss(monkeypatch, p, q, advantages, 0.1, 3.0, low=0.0, high=0.0)
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(0.0))
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), p)[0], torch.tensor([0.0, 0.0, -1.0, 1.0]))


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_bpo_dispatch_matches_dense_gradient_with_masks(monkeypatch, per_token, regularize, use_rollout_logprobs):
    args = make_args(
        loss_type="bpo",
        use_rollout_logprobs=use_rollout_logprobs,
        eps_clip_high=0.28,
        bpo_smoothing=0.1,
        bpo_weight_cap=3.0,
        true_on_policy_mode=False,
        observe_training_entropy=True,
        entropy_coef=0.03 if regularize else 0.0,
        use_kl_loss=regularize,
        kl_loss_coef=0.2,
        kl_loss_type="k2",
        calculate_per_token_loss=per_token,
    )
    logits = torch.randn((1, 6, 5), generator=torch.Generator().manual_seed(7), requires_grad=True)
    q = torch.tensor([math.log(0.7), torch.nan, math.log(0.4), math.log(0.2)], requires_grad=True)
    advantages = torch.tensor([1.0, torch.nan, -2.0, 0.0], requires_grad=True)
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1, 2, 3, 4, 0])],
        total_lengths=[6],
        response_lengths=[4],
        loss_masks=[torch.tensor([1, 0, 1, 1])],
        advantages=[advantages],
        rollout_log_probs=[q if use_rollout_logprobs else torch.full((4,), -0.5)],
        log_probs=[torch.full((4,), -0.5) if use_rollout_logprobs else q],
        ref_log_probs=[torch.full((4,), -1.5)],
    )
    reduce = get_sum_of_sample_mean([6], [4], batch["loss_masks"], per_token)
    dense_logits = logits.detach().clone().requires_grad_()
    loss, metrics = get_loss_function(args)(args, batch, logits, reduce)
    distribution = dense_logits[0, 1:5].log_softmax(-1)
    selected = distribution[torch.arange(4), torch.tensor([2, 3, 4, 0])]
    active = torch.tensor([True, False, True, True])
    coefficient = torch.zeros(4)
    masked = torch.zeros(4)
    for i in [0, 2, 3]:
        weight = (1.1 - q[i].detach().exp()) / (1.1 - selected[i].detach().exp())
        a = advantages[i].item()
        reject = (a > 0 and weight > 1.28) or (a < 0 and weight < 0.8)
        coefficient[i] = 0.0 if reject else -a * min(weight.item(), 3.0)
        masked[i] = float(reject)
    expected_pg = (coefficient[active] * selected[active]).sum() / (3 if not per_token else 1)
    expected = expected_pg
    if regularize:
        entropy = -(distribution.exp() * distribution).sum(-1)
        penalty = (selected + 1.5).square() / 2
        expected = expected + (-0.03 * entropy[active] + 0.2 * penalty[active]).sum() / (3 if not per_token else 1)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["pg_loss"], expected_pg)
    torch.testing.assert_close(metrics["pg_clipfrac"], masked[active].sum() / (3 if not per_token else 1))
    assert {
        "loss",
        "pg_loss",
        "entropy_loss",
        "pg_clipfrac",
        "ppo_kl",
        "ess_ratio",
        "train_rollout_logprob_abs_diff",
        "train_rollout_kl",
    } <= metrics.keys()
    gradient, q_gradient, a_gradient = torch.autograd.grad(loss, (logits, q, advantages), allow_unused=True)
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, dense_logits)[0], atol=1e-3, rtol=1e-2)
    assert q_gradient is None and a_gradient is None
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("per_token", [False, True])
def test_bpo_keeps_policy_loss_aggregation(monkeypatch, per_token):
    args = make_args(
        loss_type="bpo",
        use_rollout_logprobs=True,
        eps_clip_high=0.28,
        bpo_smoothing=0.1,
        bpo_weight_cap=3.0,
        entropy_coef=0.0,
        observe_training_entropy=False,
        calculate_per_token_loss=per_token,
    )
    scores = torch.tensor([0.3, 0.5, 0.2, 0.4]).log().requires_grad_()
    q = torch.tensor([0.2, 0.4, 0.3, 0.5])
    a = torch.tensor([1.0, -1.0, 0.0, 2.0])
    batch = dict(
        advantages=list(a.split([3, 1])),
        rollout_log_probs=list(q.log().split([3, 1])),
        unconcat_tokens=[torch.zeros(4, dtype=torch.long), torch.zeros(2, dtype=torch.long)],
        total_lengths=[4, 2],
        response_lengths=[3, 1],
        loss_masks=[torch.tensor([1.0, 1.0, 0.0]), torch.ones(1)],
    )
    monkeypatch.setattr(
        loss_utils, "get_log_probs_and_entropy", lambda *args, **kw: {"log_probs": list(scores.split([3, 1]))}
    )
    logits = torch.zeros(1, 6, 2)
    actual, _, _ = loss_function(args, batch, 1, logits, num_rollouts=2, num_step_tokens=torch.tensor(3))
    weight = ((1.1 - q) / (1.1 - scores.detach().exp())).clamp(max=3)
    tokens = -a * weight * scores
    expected = (tokens[0] + tokens[1] + tokens[3]) / 3 if per_token else ((tokens[0] + tokens[1]) / 2 + tokens[3]) / 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, scores, retain_graph=True)[0], torch.autograd.grad(expected, scores)[0]
    )


@pytest.mark.parametrize("use_rollout_logprobs,include_rollout", [(True, True), (False, True), (False, False)])
def test_bpo_uses_the_configured_old_policy_source(monkeypatch, use_rollout_logprobs, include_rollout):
    probabilities = [0.2, 0.6, 0.9, 0.4]
    rollout_probabilities = [0.8, 0.1, 0.4, 0.2]
    trainer_probabilities = [0.1, 0.8, 0.7, 0.9]
    scores = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    rollout_scores = torch.tensor(rollout_probabilities, dtype=torch.float64).log().requires_grad_()
    trainer_scores = torch.tensor(trainer_probabilities, dtype=torch.float64).log().requires_grad_()
    advantages = torch.tensor([1.0, -2.0, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    actual, metrics = _token_loss(
        monkeypatch,
        scores,
        rollout_scores if include_rollout else None,
        advantages,
        0.1,
        3.0,
        use_rollout_logprobs=use_rollout_logprobs,
        trainer_scores=trainer_scores,
    )
    old_probabilities = rollout_probabilities if use_rollout_logprobs else trainer_probabilities
    coefficients = []
    for p, q, advantage in zip(probabilities, old_probabilities, advantages.tolist(), strict=True):
        weight = (1.1 - q) / (1.1 - p)
        clipped = (advantage > 0 and weight > 1.28) or (advantage < 0 and weight < 0.8)
        coefficients.append(0.0 if clipped else -advantage * min(weight, 3.0))
    expected = torch.tensor(coefficients, dtype=torch.float64)
    torch.testing.assert_close(actual, (expected * scores.detach()).sum())
    gradients = torch.autograd.grad(actual, (scores, rollout_scores, trainer_scores, advantages), allow_unused=True)
    torch.testing.assert_close(gradients[0], expected)
    assert all(gradient is None for gradient in gradients[1:])
    old_scores = rollout_scores if use_rollout_logprobs else trainer_scores
    torch.testing.assert_close(metrics["ppo_kl"], (old_scores.detach() - scores.detach()).sum())


def _parse(*options):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser.parse_args(["--rollout-batch-size", "64", "--num-rollout", "1", *options])


@pytest.mark.parametrize("extra,high", [((), 0.28), (("--eps-clip-high", "0.4"), 0.4)])
@pytest.mark.parametrize("source", [(), ("--use-rollout-logprobs",)])
def test_bpo_configuration_defaults_and_overrides(extra, high, source):
    args = _parse("--loss-type", "bpo", *source, *extra)
    miles_validate_args(args)
    assert (args.eps_clip, args.eps_clip_high, args.bpo_weight_cap, args.bpo_smoothing) == (0.2, high, 3.0, 0.1)


@pytest.mark.parametrize(
    "options,message",
    [
        (("--advantage-estimator", "gspo"), "advantage-estimator grpo"),
        (("--bpo-smoothing", "0"), "bpo-smoothing"),
        (("--bpo-weight-cap", "nan"), "bpo-weight-cap"),
    ],
)
@pytest.mark.parametrize("source", [(), ("--use-rollout-logprobs",)])
def test_bpo_invalid_configuration_fails_at_startup(options, message, source):
    with pytest.raises(ValueError, match=message):
        miles_validate_args(_parse("--loss-type", "bpo", *source, *options))


def test_sc_bpo_dispatch_and_previous_loss_name_rejected():
    args = _parse("--loss-type", "sc_bpo")
    assert get_loss_function(args) is score_centering_loss_function
    with pytest.raises(SystemExit):
        _parse("--loss-type", "score_centering")
