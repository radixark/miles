"""DPPO's binary-TV policy loss (--policy-loss-type dppo_tv) against its definition, token by token."""

import math

import torch

from miles.backends.training_utils.loss.hub.math_utils import compute_dppo_tv_policy_loss


def _loss(log_pi, log_mu, adv, threshold=0.1):
    log_pi = torch.tensor(log_pi, dtype=torch.float64, requires_grad=True)
    log_mu = torch.tensor(log_mu, dtype=torch.float64)
    adv = torch.tensor(adv, dtype=torch.float64)
    pg, masked = compute_dppo_tv_policy_loss(log_mu - log_pi, log_pi, log_mu, adv, threshold=threshold)
    return pg, masked, log_pi


def test_inside_the_trust_region_it_is_minus_a_times_the_ratio():
    log_mu = [math.log(0.5), math.log(0.5), math.log(0.9)]
    log_pi = [math.log(0.55), math.log(0.45), math.log(0.95)]  # |μ - π| = 0.05 everywhere
    pg, masked, _ = _loss(log_pi, log_mu, [1.0, -1.0, 2.0])
    ratio = torch.tensor([0.55 / 0.5, 0.45 / 0.5, 0.95 / 0.9], dtype=torch.float64)
    torch.testing.assert_close(pg, -ratio * torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64))
    assert masked.sum() == 0


def test_outside_the_region_only_updates_that_push_further_away_are_masked():
    # Each token has |μ - π| = 0.3 > 0.1.
    log_mu = [math.log(0.5)] * 4
    log_pi = [math.log(0.8), math.log(0.8), math.log(0.2), math.log(0.2)]  # r > 1, r > 1, r < 1, r < 1
    adv = [1.0, -1.0, -1.0, 1.0]  # away, back, away, back
    pg, masked, log_pi_t = _loss(log_pi, log_mu, adv)
    assert masked.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert pg[0] == 0 and pg[2] == 0
    pg.sum().backward()
    grad = log_pi_t.grad
    assert grad[0] == 0 and grad[2] == 0, "a masked token gets no gradient"
    assert grad[1] != 0 and grad[3] != 0, "an update back toward the rollout policy is never masked"


def test_no_ratio_clip_unlike_ppo():
    # r = 3 with |μ - π| = 0.04 < 0.1: PPO would clip at 1.28, DPPO keeps the full ratio.
    pg, masked, _ = _loss([math.log(0.06)], [math.log(0.02)], [1.0])
    torch.testing.assert_close(pg, torch.tensor([-3.0], dtype=torch.float64))
    assert masked.sum() == 0


def test_zero_advantage_or_padding_contributes_nothing():
    pg, masked, _ = _loss([math.log(0.9), 0.0], [math.log(0.1), 0.0], [0.0, 0.0])
    torch.testing.assert_close(pg, torch.zeros(2, dtype=torch.float64))
    assert masked.sum() == 0
