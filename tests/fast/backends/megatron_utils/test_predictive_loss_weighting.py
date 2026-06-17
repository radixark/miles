import pytest
import torch

from miles.backends.megatron_utils.predictive_router_replay import (
    apply_predictive_flip_fallback,
    build_hidden_shift_loss_weights,
    build_topk_boundary_loss_weights,
    compute_hidden_shift_relative_norm,
    compute_predictive_loss,
)


def test_compute_hidden_shift_relative_norm_tracks_per_token_shift():
    old_inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    current_inputs = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

    relative_norm = compute_hidden_shift_relative_norm(
        old_inputs=old_inputs,
        current_inputs=current_inputs,
    )

    assert torch.allclose(relative_norm, torch.tensor([0.0, 0.5]))


def test_build_hidden_shift_loss_weights_masks_large_shift_tokens():
    old_inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    current_inputs = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

    weights, metrics = build_hidden_shift_loss_weights(
        old_inputs=old_inputs,
        current_inputs=current_inputs,
        max_hidden_shift_relative_norm=0.25,
        weight_mode="binary",
    )

    assert torch.equal(weights, torch.tensor([1.0, 0.0]))
    assert metrics["predictive_hidden_shift_relative_norm_mean"] == 0.25
    assert metrics["predictive_hidden_shift_relative_norm_max"] == 0.5
    assert metrics["predictive_hidden_shift_safe_fraction"] == 0.5
    assert metrics["predictive_hidden_shift_weight_mean"] == 0.5


def test_build_hidden_shift_loss_weights_supports_continuous_confidence_modes():
    old_inputs = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    current_inputs = torch.tensor([[2.0, 0.0], [2.5, 0.0], [3.0, 0.0]])

    linear_weights, linear_metrics = build_hidden_shift_loss_weights(
        old_inputs=old_inputs,
        current_inputs=current_inputs,
        max_hidden_shift_relative_norm=0.5,
        weight_mode="linear",
    )
    quadratic_weights, quadratic_metrics = build_hidden_shift_loss_weights(
        old_inputs=old_inputs,
        current_inputs=current_inputs,
        max_hidden_shift_relative_norm=0.5,
        weight_mode="quadratic",
    )

    assert torch.allclose(linear_weights, torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(quadratic_weights, torch.tensor([1.0, 0.25, 0.0]))
    assert linear_metrics["predictive_hidden_shift_safe_fraction"] == pytest.approx(2.0 / 3.0)
    assert linear_metrics["predictive_hidden_shift_weight_mean"] == pytest.approx(0.5)
    assert quadratic_metrics["predictive_hidden_shift_weight_mean"] == pytest.approx((1.0 + 0.25) / 3.0)


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


def test_build_hidden_shift_loss_weights_rejects_unknown_mode():
    old_inputs = torch.zeros(2, 4)
    current_inputs = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Unsupported hidden-shift weight mode"):
        build_hidden_shift_loss_weights(
            old_inputs=old_inputs,
            current_inputs=current_inputs,
            max_hidden_shift_relative_norm=0.5,
            weight_mode="not-a-mode",
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


def test_apply_predictive_flip_fallback_reverts_low_margin_route_changes():
    old_logits = torch.tensor(
        [
            [4.0, 3.95, 0.0],
            [4.0, 1.0, 0.0],
        ]
    )
    adjusted_logits = torch.tensor(
        [
            [3.94, 3.96, 0.0],
            [0.9, 1.1, 0.0],
        ]
    )

    effective_logits, applied_delta_logits, execution_weights, metrics = apply_predictive_flip_fallback(
        reference_logits=old_logits,
        adjusted_logits=adjusted_logits,
        topk=1,
        min_post_topk_margin_for_flip=0.08,
    )

    assert torch.allclose(effective_logits[0], old_logits[0])
    assert torch.allclose(effective_logits[1], adjusted_logits[1])
    assert torch.allclose(applied_delta_logits[0], torch.zeros_like(applied_delta_logits[0]))
    assert execution_weights.tolist() == [0.0, 1.0]
    assert metrics["predictive_route_change_fraction"] == pytest.approx(1.0)
    assert metrics["predictive_flip_fallback_fraction"] == pytest.approx(0.5)
    assert metrics["predictive_confident_flip_fraction"] == pytest.approx(0.5)


def test_apply_predictive_flip_fallback_reports_applied_delta_without_threshold():
    old_logits = torch.tensor([[4.0, 3.9, 0.0]])
    adjusted_logits = torch.tensor([[3.8, 4.1, 0.0]])

    effective_logits, applied_delta_logits, execution_weights, metrics = apply_predictive_flip_fallback(
        reference_logits=old_logits,
        adjusted_logits=adjusted_logits,
        topk=1,
        min_post_topk_margin_for_flip=None,
    )

    assert torch.allclose(effective_logits, adjusted_logits)
    assert torch.allclose(applied_delta_logits, adjusted_logits - old_logits)
    assert torch.allclose(execution_weights, torch.ones_like(execution_weights))
    assert metrics["predictive_flip_fallback_fraction"] == 0.0
    assert metrics["predictive_confident_flip_fraction"] == pytest.approx(1.0)
