from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss_hub.candidate_opd_ops import candidate_policy_loss, selected_log_softmax
from miles.rollout.opd_balance import set_domain_weights
from miles.utils.types import Sample


def test_selected_log_softmax_matches_dense_forward_and_gradient():
    torch.manual_seed(71)
    logits = torch.randn(7, 19, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    ids = torch.stack([torch.randperm(19)[:4] for _ in range(7)])
    upstream = torch.randn(7, 4)
    result = selected_log_softmax(logits, ids)
    expected = oracle.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(result, expected)
    (result * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(logits.grad, oracle.grad)


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("dual_clip", [None, 3.0])
def test_candidate_ppo_matches_independent_dense_gradient(refresh, dual_clip):
    torch.manual_seed(3)
    logits = torch.randn(5, 13, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ids = torch.stack([torch.randperm(13)[:4] for _ in range(5)])
    old = torch.randn(5, 13).log_softmax(-1).gather(-1, ids).requires_grad_()
    teacher = torch.randn(5, 13).log_softmax(-1).gather(-1, ids).requires_grad_()
    current = selected_log_softmax(logits, ids)
    loss, _, _ = candidate_policy_loss(
        current, old, teacher, refresh=refresh, eps_low=0.2, eps_high=0.25, dual_clip=dual_clip
    )
    dense = reference.log_softmax(-1)
    objective = reference.new_zeros(())
    # Independent scalar oracle: clipping occurs for each candidate before summing.
    for row in range(5):
        reward_q = dense[row, ids[row]].detach() if refresh else old[row].detach()
        weights = reward_q.exp() / reward_q.exp().sum()
        for col in range(4):
            advantage = (weights[col] * (teacher[row, col].detach() - reward_q[col])).detach()
            ratio = (dense[row, ids[row, col]] - old[row, col].detach()).clamp(-20, 20).exp()
            term = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.25) * advantage)
            if dual_clip is not None and advantage < 0:
                term = torch.maximum(term, dual_clip * advantage)
            objective = objective - term
    loss.sum().backward()
    objective.backward()
    torch.testing.assert_close(loss.sum(), objective)
    torch.testing.assert_close(logits.grad, reference.grad)
    assert old.grad is None and teacher.grad is None


def test_refresh_keeps_old_denominator_fixed():
    current = torch.tensor([[-0.1, -2.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -0.5]])
    teacher = torch.tensor([[-0.2, -1.7]])
    fixed, _, clips = candidate_policy_loss(current, old, teacher, refresh=True, eps_low=0.2, eps_high=0.2)
    wrong, _, _ = candidate_policy_loss(current, current.detach(), teacher, refresh=True, eps_low=0.2, eps_high=0.2)
    assert not torch.allclose(fixed, wrong)
    assert clips.item() == 1


def _sample(domain, length, gap=1):
    return Sample(
        response_length=length,
        metadata={"domain": domain},
        opd_candidate_old_log_probs=torch.full((length, 2), -1.0),
        opd_candidate_teacher_log_probs=torch.full((length, 2), -1.0 - gap),
    )


@pytest.mark.parametrize("token_mean", [False, True])
def test_domain_balance_corrects_actual_reduction_mass(token_mean):
    samples = [_sample("a", 2), _sample("a", 4), _sample("b", 10)]
    args = Namespace(
        opd_domain_balance="static", opd_domain_targets=["a=0.5", "b=0.5"], calculate_per_token_loss=token_mean
    )
    set_domain_weights(args, samples)
    masses = [sample.response_length if token_mean else 1 for sample in samples]
    weighted = [s.opd_loss_weights[0].item() * m for s, m in zip(samples, masses, strict=True)]
    assert sum(weighted[:2]) == pytest.approx(weighted[2])
    assert sum(weighted) == pytest.approx(sum(masses))


def test_zero_gap_falls_back_to_static_and_missing_domain_fails():
    args = Namespace(
        opd_domain_balance="gap",
        opd_domain_targets=["a=0.5", "b=0.5"],
        calculate_per_token_loss=False,
        opd_gap_alpha=1,
    )
    samples = [_sample("a", 2, gap=0), _sample("b", 4, gap=0)]
    set_domain_weights(args, samples)
    assert all(torch.equal(s.opd_loss_weights, torch.ones(s.response_length)) for s in samples)
    with pytest.raises(ValueError, match="Every configured"):
        set_domain_weights(args, samples[:1])


def test_empty_response_has_differentiable_zero_loss():
    logits = torch.empty(0, 17, requires_grad=True)
    current = selected_log_softmax(logits, torch.empty(0, 4, dtype=torch.long))
    loss, _, _ = candidate_policy_loss(
        current, current.detach(), current.detach(), refresh=True, eps_low=0.2, eps_high=0.2
    )
    loss.sum().backward()
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("dual_clip", [None, 3.0])
def test_candidate_ratio_remains_finite_for_extreme_tail_mismatch(dual_clip):
    current = torch.tensor([[-1.0, -0.5]], requires_grad=True)
    old = torch.tensor([[-1000.0, -0.5]])
    teacher = torch.tensor([[-2.0, -0.7]])
    loss, _, _ = candidate_policy_loss(
        current, old, teacher, refresh=True, eps_low=0.2, eps_high=0.2, dual_clip=dual_clip
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(current.grad).all()
    if dual_clip is not None:
        assert loss.item() < 3.0
