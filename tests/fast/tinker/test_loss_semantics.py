import torch

from miles.backends.training_utils.loss_hub.math_utils import (
    compute_importance_sampling_loss,
    compute_policy_loss,
)


def test_importance_sampling_is_not_ppo_clipped() -> None:
    # new/old probability ratio = exp(1), beyond PPO's 1.2 upper clip.
    ppo_kl = torch.tensor([-1.0])
    advantages = torch.tensor([1.0])

    is_loss, is_clipfrac = compute_importance_sampling_loss(ppo_kl, advantages)
    ppo_loss, ppo_clipfrac = compute_policy_loss(ppo_kl, advantages, 0.2, 0.2)

    torch.testing.assert_close(is_loss, torch.tensor([-torch.e]))
    torch.testing.assert_close(ppo_loss, torch.tensor([-1.2]))
    torch.testing.assert_close(is_clipfrac, torch.zeros(1))
    torch.testing.assert_close(ppo_clipfrac, torch.ones(1))
