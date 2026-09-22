from __future__ import annotations

import torch

from miles.backends.training_utils.loss_hub.math_utils import compute_top_entropy_mask


def test_keeps_the_top_share_of_active_tokens_by_entropy():
    entropy = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.3, 0.8])
    loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    mask = compute_top_entropy_mask(entropy, loss_mask, quantile=0.4)

    # Five active tokens; the 0.6 quantile of {0.1, 0.9, 0.5, 0.7, 0.3} is 0.58, so
    # 0.9 and 0.7 survive. The masked-out 0.8 never counts, however high it is.
    assert mask.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert mask.dtype == entropy.dtype


def test_quantile_one_returns_the_loss_mask():
    entropy = torch.tensor([0.1, 0.9, 0.5])
    loss_mask = torch.tensor([1.0, 0.0, 1.0])

    assert compute_top_entropy_mask(entropy, loss_mask, quantile=1.0).tolist() == [1.0, 0.0, 1.0]


def test_all_inactive_tokens_stay_masked_without_quantile_error():
    entropy = torch.tensor([0.1, 0.9])
    loss_mask = torch.zeros(2)

    assert compute_top_entropy_mask(entropy, loss_mask, quantile=0.2).tolist() == [0.0, 0.0]


def test_mask_does_not_carry_gradient():
    entropy = torch.tensor([0.1, 0.9, 0.5], requires_grad=True)
    loss_mask = torch.ones(3)

    mask = compute_top_entropy_mask(entropy, loss_mask, quantile=0.5)

    assert not mask.requires_grad
