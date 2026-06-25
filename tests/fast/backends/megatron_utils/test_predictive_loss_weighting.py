import pytest
import torch

from miles.backends.megatron_utils.predictive_router_replay import (
    build_topk_boundary_loss_weights,
    compute_predictive_loss,
)


def test_compute_predictive_loss_respects_sample_weights():
    old_logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    current_logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    predicted_delta_logits = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)
    sample_weights = torch.tensor([1.0, 0.0])

    weighted_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="kl",
        sample_weights=sample_weights,
    )
    unweighted_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits.detach(),
        loss_type="kl",
        sample_weights=None,
    )

    assert weighted_loss.item() == pytest.approx(0.0)
    assert unweighted_loss.item() > 0.0

    weighted_loss.backward()
    assert torch.allclose(predicted_delta_logits.grad[0], torch.zeros_like(predicted_delta_logits.grad[0]))
    assert torch.allclose(predicted_delta_logits.grad[1], torch.zeros_like(predicted_delta_logits.grad[1]))


def test_compute_predictive_loss_rejects_unknown_loss_type():
    old_logits = torch.zeros(2, 4)
    current_logits = torch.ones(2, 4)
    predicted_delta_logits = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Unsupported predictive loss type"):
        compute_predictive_loss(
            old_logits=old_logits,
            current_logits=current_logits,
            predicted_delta_logits=predicted_delta_logits,
            loss_type="l2",  # removed in this PR, must error
        )


def test_compute_predictive_loss_rejects_shape_mismatch():
    old_logits = torch.zeros(2, 4)
    current_logits = torch.ones(2, 4)
    bad_predicted = torch.ones(2, 8)  # last-dim mismatch
    with pytest.raises(ValueError, match="predicted_delta_logits shape"):
        compute_predictive_loss(
            old_logits=old_logits,
            current_logits=current_logits,
            predicted_delta_logits=bad_predicted,
            loss_type="kl-post",
        )


def test_build_topk_boundary_loss_weights_emphasizes_small_margin_tokens():
    old_logits = torch.tensor(
        [
            [5.0, 4.9, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )

    weights, metrics = build_topk_boundary_loss_weights(
        old_logits=old_logits,
        topk=1,
        max_boundary_loss_weight=4.0,
        min_boundary_margin=1e-4,
    )

    assert weights is not None
    assert weights.shape == torch.Size([2])
    assert weights[0].item() > weights[1].item()
    assert metrics["predictive_boundary_loss_weight_max"] >= metrics["predictive_boundary_loss_weight_mean"]
    assert metrics["predictive_boundary_margin_min"] == 0.09999990463256836
