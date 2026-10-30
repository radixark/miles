from argparse import Namespace

import pytest

from miles.rollout.reward_shaping import apply_length_penalty, reward_group_segments, validate_length_penalty_args
from miles.utils.types import Sample


def _args(**overrides) -> Namespace:
    return Namespace(
        **{
            "length_penalty_coeff": 1.0,
            "length_penalty_min_tokens": 1,
            "custom_reward_post_process_path": None,
            "custom_convert_samples_to_train_data_path": None,
            "reward_key": None,
            "n_samples_per_prompt": 4,
            "rollout_batch_size": 2,
            **overrides,
        }
    )


def _shape(args: Namespace, samples: list[Sample]) -> list[float]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    return apply_length_penalty(args, samples, rewards, reward_group_segments(args, samples))


@pytest.mark.parametrize("coeff, expected", [(0.0, 1.0), (0.5, 2**-0.5), (1.0, 0.5), (2.0, 0.25)])
def test_correct_only_and_configurable_exponent(coeff: float, expected: float) -> None:
    samples = [
        Sample(group_index=0, response_length=100, reward=1.0),
        Sample(group_index=0, response_length=200, reward=1.0),
        Sample(group_index=0, response_length=1, reward=0.0),
        Sample(group_index=0, response_length=1000, reward=-0.5),
        Sample(group_index=0, response_length=1000, reward=0.9),
    ]
    assert _shape(_args(length_penalty_coeff=coeff), samples) == pytest.approx([1, expected, 0, -0.5, 0.9])
    assert [sample.reward for sample in samples] == [1.0, 1.0, 0.0, -0.5, 0.9]


def test_minimum_protects_zero_and_short_trajectories() -> None:
    samples = [Sample(group_index=0, response_length=length, reward=1.0) for length in [0, 20, 100, 200]]
    assert _shape(_args(length_penalty_min_tokens=100), samples) == [1, 1, 1, 0.5]
    assert _shape(_args(), [samples[0], samples[3]]) == [1, 1]


def test_ties_single_correct_and_no_correct() -> None:
    samples = [
        Sample(group_index=0, response_length=10, reward=1),
        Sample(group_index=1, response_length=1, reward=0),
        Sample(group_index=0, response_length=10, reward=1),
        Sample(group_index=1, response_length=100, reward=-1),
        Sample(group_index=2, response_length=200, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 0, 1, -1, 1]
    assert _shape(_args(), []) == []


def test_interleaved_prompt_groups_and_rollout_segments() -> None:
    samples = [
        Sample(group_index=0, rollout_id=10, response_length=40, reward=1),
        Sample(group_index=1, rollout_id=20, response_length=200, reward=1),
        Sample(group_index=0, rollout_id=11, response_length=200, reward=1),
        Sample(group_index=0, rollout_id=10, response_length=60, reward=1),
        Sample(group_index=1, rollout_id=21, response_length=400, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 1, 0.5, 1, 0.5]


def test_explicit_prompt_boundaries_override_indices() -> None:
    samples = [Sample(group_index=0, response_length=length, reward=1) for length in [10, 20, 100, 400]]
    rewards = apply_length_penalty(_args(), samples, [1] * 4, reward_group_segments(_args(), samples, [2, 2]))
    assert rewards == [1, 0.5, 1, 0.25]


def test_only_loss_mask_tokens_count() -> None:
    samples = [
        Sample(group_index=0, response_length=1000, loss_mask=[1] * 100 + [0] * 900, reward=1),
        Sample(group_index=0, response_length=2000, loss_mask=[1] * 200 + [0] * 1800, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 0.5]

    # Tokens masked off when resuming a partial rollout are not penalized.
    samples[1].loss_mask = [1] * 50 + [0] * 1950
    assert _shape(_args(), samples) == [0.5, 1]


def test_removed_samples_contribute_no_tokens() -> None:
    samples = [
        Sample(group_index=0, rollout_id=1, response_length=100, reward=1),
        Sample(group_index=0, rollout_id=1, response_length=1000, remove_sample=True, reward=1),
        Sample(group_index=0, rollout_id=2, response_length=200, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 1, 0.5]


@pytest.mark.parametrize("removed", [False, True])
def test_zero_trainable_rollout_does_not_set_reference(removed: bool) -> None:
    samples = [
        Sample(group_index=0, rollout_id=1, response_length=100, loss_mask=[0] * 100, remove_sample=removed, reward=1),
        Sample(group_index=0, rollout_id=2, response_length=100, reward=1),
        Sample(group_index=0, rollout_id=3, response_length=200, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 1, 0.5]
    assert _shape(_args(), samples[:1]) == [1]


def test_dict_reward_and_inconsistent_siblings() -> None:
    samples = [
        Sample(group_index=0, rollout_id=1, response_length=10, reward={"accuracy": 1}),
        Sample(group_index=0, rollout_id=2, response_length=20, reward={"accuracy": 1}),
    ]
    assert _shape(_args(reward_key="accuracy"), samples) == [1, 0.5]
    samples[1].rollout_id = 1
    samples[1].reward["accuracy"] = 0
    with pytest.raises(ValueError, match="share one reward"):
        _shape(_args(reward_key="accuracy"), samples)


@pytest.mark.parametrize("coeff", [-1, float("inf"), float("nan")])
def test_invalid_coefficients(coeff: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_length_penalty_args(_args(length_penalty_coeff=coeff))


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_minimum(minimum: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_length_penalty_args(_args(length_penalty_min_tokens=minimum))


@pytest.mark.parametrize("field", ["custom_reward_post_process_path", "custom_convert_samples_to_train_data_path"])
def test_custom_overrides_cannot_silently_bypass_shaping(field: str) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_length_penalty_args(_args(**{field: "custom.function"}))
    validate_length_penalty_args(_args(length_penalty_coeff=0, **{field: "custom.function"}))
