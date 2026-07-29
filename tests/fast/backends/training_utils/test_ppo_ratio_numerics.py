import torch

from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl, compute_policy_loss

compute_approx_kl_eager = compute_approx_kl.__wrapped__
compute_policy_loss_eager = compute_policy_loss.__wrapped__


def test_policy_loss_extreme_log_ratios_with_zero_advantages_stay_finite():
    ppo_kl = torch.tensor([-1000.0, 1000.0, float("nan"), float("inf"), float("-inf")])
    advantages = torch.zeros_like(ppo_kl)

    pg_losses, clipfrac = compute_policy_loss_eager(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    assert torch.isfinite(pg_losses).all().item()
    assert torch.isfinite(clipfrac).all().item()
    torch.testing.assert_close(pg_losses, torch.zeros_like(pg_losses))


def test_policy_loss_matches_unclamped_ratio_for_normal_log_ratios():
    ppo_kl = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)
    advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    eps_clip = 0.2
    eps_clip_high = 0.2

    pg_losses, clipfrac = compute_policy_loss_eager(
        ppo_kl,
        advantages,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )

    ratio = (-ppo_kl).exp()
    expected_losses1 = -ratio * advantages
    expected_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    expected_losses = torch.maximum(expected_losses1, expected_losses2)
    expected_clipfrac = torch.gt(expected_losses2, expected_losses1).float()

    torch.testing.assert_close(pg_losses, expected_losses)
    torch.testing.assert_close(clipfrac, expected_clipfrac)


def test_low_var_kl_extreme_log_ratios_stay_finite():
    log_probs = torch.tensor([-1000.0, 1000.0, float("nan"), float("inf"), float("-inf")])
    log_probs_base = torch.zeros_like(log_probs)

    kl = compute_approx_kl_eager(log_probs, log_probs_base, kl_loss_type="low_var_kl")

    assert torch.isfinite(kl).all().item()


def test_low_var_kl_matches_unclamped_formula_for_normal_log_ratios():
    log_probs = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)
    log_probs_base = torch.zeros_like(log_probs)

    kl = compute_approx_kl_eager(log_probs, log_probs_base, kl_loss_type="low_var_kl")

    log_ratio = -(log_probs - log_probs_base)
    expected_kl = log_ratio.exp() - 1 - log_ratio
    torch.testing.assert_close(kl, expected_kl)
