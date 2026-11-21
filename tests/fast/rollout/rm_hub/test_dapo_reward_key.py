"""`--rm-type dapo` grades to a dict; without `--reward-key` the run must still train on its score."""

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.rm_hub import async_rm
from miles.utils.arguments import default_reward_key_for_dict_rewards

# two prompts x two samples: one right and one wrong answer each
RESPONSES = ["Answer: 4", "Answer: 5", "Answer: 7", "Answer: 6"]
LABELS = ["4", "4", "7", "7"]


def _graded_samples(rm_type: str):
    args = make_args(rm_type=rm_type, custom_rm_path=None, reward_key=None)
    samples = make_samples_grouped(2, 2)
    for sample, response, label in zip(samples, RESPONSES, LABELS, strict=True):
        sample.response = response if rm_type == "dapo" else f"\\boxed{{{response.removeprefix('Answer: ')}}}"
        sample.label = label
        sample.reward = asyncio.run(async_rm(args, sample))
    return samples


def _training_args(**overrides):
    return make_args(
        custom_rm_path=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        **overrides,
    )


@pytest.mark.parametrize("rm_type", ["dapo", "boxed_dapo"])
def test_dapo_without_reward_key_trains_on_the_score(rm_type):
    samples = _graded_samples(rm_type)
    assert all(isinstance(sample.reward, dict) for sample in samples)

    args = _training_args(rm_type=rm_type)
    default_reward_key_for_dict_rewards(args)

    assert args.reward_key == "score"
    raw, normalized = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
    assert raw == [1.0, -1.0, 1.0, -1.0]
    assert normalized == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_without_the_default_dict_rewards_never_reach_training():
    """The failure the default prevents: the first rollout's rewards cannot become a tensor."""
    samples = _graded_samples("dapo")
    args = _training_args(rm_type="dapo")
    with pytest.raises(TypeError, match="dict"):
        _post_process_rewards(args, samples, custom_reward_post_process_func=None)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"rm_type": "dapo", "reward_key": "acc"}, "acc"),  # an explicit key wins
        ({"rm_type": "dapo", "custom_rm_path": "my_pkg.rm"}, None),  # a custom rm decides its own reward shape
        ({"rm_type": "math"}, None),  # graders that return a number need no key
        ({"rm_type": None}, None),
    ],
)
def test_default_applies_only_to_builtin_dict_graders(overrides, expected):
    args = make_args(**{"custom_rm_path": None, "reward_key": None, **overrides})
    default_reward_key_for_dict_rewards(args)
    assert args.reward_key == expected
