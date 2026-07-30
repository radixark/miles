import pytest
import torch

from miles.backends.megatron_utils.tinker_loss import (
    compute_tinker_loss,
    tensor_data,
    validate_and_get_targets,
)
from miles.ray.tinker.protocol import TinkerError


def _rl_inputs():
    return {
        "target_tokens": torch.tensor([1, 2]),
        "logprobs": torch.tensor([-0.4, -0.7]),
        "advantages": torch.tensor([2.0, -1.0]),
    }


def test_cross_entropy_topk_is_sum_reduced():
    target_logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], requires_grad=True)
    inputs = {
        "target_tokens": torch.tensor([[1, 2], [3, 4]]),
        "weights": torch.tensor([[1.0, 0.5], [0.0, 2.0]]),
    }

    loss = compute_tinker_loss(
        target_logprobs,
        inputs,
        loss_fn="cross_entropy",
        loss_fn_config={},
    )

    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    torch.testing.assert_close(target_logprobs.grad, -inputs["weights"])


def test_importance_sampling_matches_public_formula():
    target_logprobs = torch.tensor([-0.2, -1.0])
    inputs = _rl_inputs()

    actual = compute_tinker_loss(
        target_logprobs,
        inputs,
        loss_fn="importance_sampling",
        loss_fn_config={},
    )
    expected = -(torch.exp(target_logprobs - inputs["logprobs"]) * inputs["advantages"]).sum()
    torch.testing.assert_close(actual, expected)


def test_ppo_uses_ratio_thresholds():
    target_logprobs = torch.tensor([0.0, 0.0])
    inputs = {
        "target_tokens": torch.tensor([1, 2]),
        "logprobs": torch.log(torch.tensor([0.5, 2.0])),
        "advantages": torch.tensor([1.0, -1.0]),
    }

    actual = compute_tinker_loss(
        target_logprobs,
        inputs,
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    )

    ratio = torch.exp(target_logprobs - inputs["logprobs"])
    expected = -torch.minimum(
        ratio * inputs["advantages"],
        ratio.clamp(0.8, 1.2) * inputs["advantages"],
    ).sum()
    torch.testing.assert_close(actual, expected)


def test_cispo_detaches_clipped_ratio():
    target_logprobs = torch.tensor([-0.1, -0.2], requires_grad=True)
    inputs = _rl_inputs()

    loss = compute_tinker_loss(
        target_logprobs,
        inputs,
        loss_fn="cispo",
        loss_fn_config={},
    )
    coefficient = torch.exp(target_logprobs.detach() - inputs["logprobs"]).clamp(0.0, 4.0)
    loss.backward()

    torch.testing.assert_close(target_logprobs.grad, -(coefficient * inputs["advantages"]))


def test_dro_matches_public_formula():
    target_logprobs = torch.tensor([-0.2, -1.0])
    inputs = _rl_inputs()

    actual = compute_tinker_loss(
        target_logprobs,
        inputs,
        loss_fn="dro",
        loss_fn_config={"beta": 0.05},
    )
    delta = target_logprobs - inputs["logprobs"]
    expected = -(target_logprobs * inputs["advantages"] - 0.5 * 0.05 * delta.square()).sum()
    torch.testing.assert_close(actual, expected)


def test_target_shape_must_match_model_input():
    with pytest.raises(TinkerError, match="must equal model_input length"):
        validate_and_get_targets(
            {
                "target_tokens": torch.tensor([1, 2]),
                "weights": torch.ones(2),
            },
            model_input_length=3,
            loss_fn="cross_entropy",
        )


def test_cross_entropy_topk_dimension_must_be_positive():
    with pytest.raises(TinkerError, match="top-K dimension must be positive"):
        validate_and_get_targets(
            {
                "target_tokens": torch.empty((2, 0), dtype=torch.int64),
                "weights": torch.empty((2, 0)),
            },
            model_input_length=2,
            loss_fn="cross_entropy",
        )


@pytest.mark.parametrize(
    ("target_logprobs", "inputs", "config", "category"),
    [
        (
            torch.tensor([float("nan"), -0.2]),
            {
                "target_tokens": torch.tensor([1, 2]),
                "weights": torch.ones(2),
            },
            {},
            "server",
        ),
        (
            torch.tensor([-0.1, -0.2]),
            {
                "target_tokens": torch.tensor([1, 2]),
                "weights": torch.tensor([1.0, float("inf")]),
            },
            {},
            "user",
        ),
        (
            torch.tensor([-0.1, -0.2]),
            {
                "target_tokens": torch.tensor([1, 2]),
                "weights": torch.ones(2),
            },
            {"clip_low_threshold": float("nan")},
            "user",
        ),
    ],
)
def test_nonfinite_loss_values_are_rejected(target_logprobs, inputs, config, category):
    loss_fn = "ppo" if config else "cross_entropy"
    if loss_fn == "ppo":
        inputs = _rl_inputs()

    with pytest.raises(TinkerError) as exc_info:
        compute_tinker_loss(
            target_logprobs,
            inputs,
            loss_fn=loss_fn,
            loss_fn_config=config,
        )

    assert exc_info.value.category == category


def test_tensor_data_serializes_shape_and_float32():
    assert tensor_data(torch.tensor([[1.0, 2.0]], dtype=torch.float64)) == {
        "data": [1.0, 2.0],
        "dtype": "float32",
        "shape": [1, 2],
    }
