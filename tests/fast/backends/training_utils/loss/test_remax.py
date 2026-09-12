import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils.loss_hub.advantages import compute_advantages


@pytest.mark.parametrize("local_lengths", [[3, 1, 4], [0, 2, 0]])
def test_remax_broadcasts_baseline_adjusted_rewards_without_normalization(local_lengths):
    args = make_args(advantage_estimator="remax", kl_coef=0.0)
    raw_rewards = [1.0, 5.0, 3.0]
    fake_baseline = 3.0
    kl = [torch.zeros(length) for length in local_lengths]
    advantages, returns = compute_advantages(
        args,
        kl=kl,
        rewards=[reward - fake_baseline for reward in raw_rewards],
        log_probs=kl,
        loss_masks=[torch.ones(4) for _ in raw_rewards],
        total_lengths=[6, 6, 6],
        response_lengths=[4, 4, 4],
    )

    for advantage, ret, length, expected in zip(advantages, returns, local_lengths, [-2.0, 2.0, 0.0], strict=True):
        torch.testing.assert_close(advantage, torch.full((length,), expected))
        torch.testing.assert_close(ret, advantage)
