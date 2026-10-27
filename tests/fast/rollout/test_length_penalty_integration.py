from argparse import ArgumentParser, Namespace

import pytest

from miles.ray.rollout.train_data_conversion import _post_process_rewards, convert_samples_to_train_data
from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
from miles.rollout.reward_shaping import validate_length_penalty_args
from miles.utils.args.configs.reward_model import RewardModelRolloutOnlyConfig
from miles.utils.types import Sample


def _args(**overrides) -> Namespace:
    return Namespace(
        **{
            "length_penalty_coeff": 1.0,
            "length_penalty_min_tokens": 1,
            "reward_key": None,
            "n_samples_per_prompt": 4,
            "rollout_batch_size": 2,
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "grpo_std_normalization": False,
            "enable_sample_ownership_checker": False,
            "rollout_top_logprobs_num": 0,
            "use_dynamic_global_batch_size": False,
            "use_rollout_routing_replay": False,
            "use_rollout_indexer_replay": False,
            **overrides,
        }
    )


def test_dynamic_filter_keeps_all_correct_groups_with_length_signal() -> None:
    samples = [Sample(group_index=0, index=i, response_length=length, reward=1) for i, length in enumerate([10, 20])]
    assert apply_reward_nonzero_std_filter(_args(), samples).keep
    assert not apply_reward_nonzero_std_filter(_args(length_penalty_coeff=0), samples).keep
    samples[1].response_length = 10
    assert not apply_reward_nonzero_std_filter(_args(), samples).keep


@pytest.mark.parametrize("normalize", [False, True])
def test_shaping_reaches_training_data_before_normalization(normalize: bool) -> None:
    samples = [
        Sample(index=i, group_index=0, tokens=[0] * length, response_length=length, reward=1.0)
        for i, length in enumerate([10, 20])
    ]
    out = convert_samples_to_train_data(
        _args(rewards_normalization=normalize),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    assert out["raw_reward"] == [1.0, 1.0]
    assert out["rewards"] == pytest.approx([0.25, -0.25] if normalize else [1.0, 0.5])
    assert [sample.reward for sample in samples] == [1.0, 1.0]


@pytest.mark.parametrize("std_normalize", [False, True])
def test_normalization_counts_each_rollout_once(std_normalize: bool) -> None:
    samples = [
        Sample(index=0, rollout_id=10, group_index=0, response_length=4, reward=1.0),
        Sample(index=1, rollout_id=10, group_index=0, response_length=6, reward=1.0),
        Sample(index=2, rollout_id=11, group_index=0, response_length=20, reward=1.0),
    ]
    raw, rewards = _post_process_rewards(_args(grpo_std_normalization=std_normalize), samples, None)
    positive = 0.25
    assert raw == [1.0, 1.0, 1.0]
    assert rewards == pytest.approx([positive, positive, -positive])


@pytest.mark.parametrize("estimator", ["grpo", "gspo"])
@pytest.mark.parametrize("coeff", [0.01, 0.1, 1.0])
def test_tiny_length_differences_keep_coefficient_scale(estimator: str, coeff: float) -> None:
    samples = [
        Sample(index=i, group_index=0, response_length=length, reward=1.0) for i, length in enumerate([1000, 1001])
    ]
    raw, rewards = _post_process_rewards(
        _args(advantage_estimator=estimator, grpo_std_normalization=True, length_penalty_coeff=coeff), samples, None
    )
    delta = (1 - 1.001 ** (-coeff)) / 2
    assert raw == [1.0, 1.0]
    assert rewards == pytest.approx([delta, -delta], abs=1e-7)


def test_mixed_groups_retain_std_normalization() -> None:
    samples = [
        Sample(index=i, group_index=0, response_length=length, reward=reward)
        for i, (length, reward) in enumerate([(1000, 1.0), (2000, 1.0), (1000, 0.0)])
    ]
    _, rewards = _post_process_rewards(_args(grpo_std_normalization=True), samples, None)
    assert rewards == pytest.approx([0.5 / (0.5 + 1e-6), 0, -0.5 / (0.5 + 1e-6)])


def test_filter_drops_negligible_length_only_spread() -> None:
    samples = [
        Sample(index=i, group_index=0, response_length=length, reward=1.0) for i, length in enumerate([1000, 1001])
    ]
    assert not apply_reward_nonzero_std_filter(_args(length_penalty_coeff=0.0001), samples).keep
    assert apply_reward_nonzero_std_filter(_args(length_penalty_coeff=0.01), samples).keep


def test_zero_coefficient_disables_shaping_in_training() -> None:
    samples = [Sample(group_index=0, index=i, response_length=length, reward=1.0) for i, length in enumerate([0, 100])]
    raw, rewards = _post_process_rewards(_args(length_penalty_coeff=0, rewards_normalization=False), samples, None)
    assert raw == rewards == [1.0, 1.0]


def test_cli_defaults_and_explicit_penalty_settings() -> None:
    parser = ArgumentParser()
    RewardModelRolloutOnlyConfig.add_arguments(parser)
    args = parser.parse_args([])
    assert (args.length_penalty_coeff, args.length_penalty_min_tokens) == (0.0, 1)
    args = parser.parse_args(["--length-penalty-coeff", "0.1", "--length-penalty-min-tokens", "1000"])
    validate_length_penalty_args(args)
    assert (args.length_penalty_coeff, args.length_penalty_min_tokens) == (0.1, 1000)


def test_penalty_uses_final_training_masks_and_excludes_removed_siblings() -> None:
    samples = [
        Sample(
            index=0,
            rollout_id=10,
            group_index=0,
            tokens=[0] * 1000,
            response_length=1000,
            loss_mask=[1] * 100 + [0] * 900,
            reward=1.0,
        ),
        Sample(
            index=1,
            rollout_id=10,
            group_index=0,
            tokens=[0] * 1000,
            response_length=1000,
            remove_sample=True,
            reward=1.0,
        ),
        Sample(
            index=2,
            rollout_id=11,
            group_index=0,
            tokens=[0] * 1000,
            response_length=1000,
            loss_mask=[0] * 950 + [1] * 50,
            reward=1.0,
        ),
    ]
    args = _args(rewards_normalization=False)
    assert apply_reward_nonzero_std_filter(args, samples).keep
    out = convert_samples_to_train_data(args, samples, {}, None, None)
    assert out["rewards"] == [0.5, 0.5, 1.0]
    assert [sum(mask) for mask in out["loss_masks"]] == [100, 0, 50]
    assert out["rollout_mask_sums"] == [100, 100, 50]


def test_partial_rollout_masked_prefix_is_not_penalized() -> None:
    samples = [
        Sample(index=0, group_index=0, tokens=[0] * 1000, response_length=1000, reward=1.0),
        Sample(
            index=1,
            group_index=0,
            tokens=[0] * 1500,
            response_length=1500,
            loss_mask=[0] * 1000 + [1] * 500,
            reward=1.0,
        ),
    ]
    out = convert_samples_to_train_data(_args(rewards_normalization=False), samples, {}, None, None)
    assert out["rewards"] == [0.5, 1.0]
    assert [sum(mask) for mask in out["loss_masks"]] == [1000, 500]
