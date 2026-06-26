"""CPU regression test for KL loss-masking in REINFORCE++ returns."""

import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import get_reinforce_plus_plus_returns
from miles.backends.training_utils.parallel import set_parallel_state

from .loss_test_utils import make_parallel_state


def test_reinforce_pp_returns_gates_kl_by_loss_mask():
    # Masked tokens (positions 1,2) carry KL; without gating their KL leaks backward
    # into the valid leading token, giving G_0 == -3.439 instead of the gated -1.729.
    set_parallel_state(make_parallel_state())
    returns = get_reinforce_plus_plus_returns(
        rewards=torch.tensor([0.0]),
        kl=[torch.tensor([1.0, 1.0, 1.0, 1.0])],
        loss_masks=[torch.tensor([1.0, 0.0, 0.0, 1.0])],
        response_lengths=[4],
        total_lengths=[4],
        kl_coef=1.0,
        gamma=0.9,
    )
    assert returns[0][0].item() == pytest.approx(-1.729)
