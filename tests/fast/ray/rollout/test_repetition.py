from argparse import Namespace

import pytest

from miles.ray.rollout.repetition import apply_repetition_reward_penalty
from miles.utils.types import Sample


def _sample(*, rollout_id: int, reward: float, response: str, index: int) -> Sample:
    return Sample(
        group_index=0,
        index=index,
        rollout_id=rollout_id,
        reward=reward,
        response=response,
        response_length=1,
        tokens=[1],
        metadata={"raw_reward": reward},
    )


def test_penalizes_repetitive_sample() -> None:
    args = Namespace(repetition_reward_penalty=0.1, reward_key=None)
    repeated = _sample(rollout_id=1, reward=0.0, response="x" * 10_000, index=1)
    normal = _sample(rollout_id=2, reward=1.0, response="short response", index=2)

    apply_repetition_reward_penalty(args, [repeated, normal])

    assert repeated.reward == pytest.approx(-0.1)
    assert repeated.metadata["raw_reward"] == pytest.approx(-0.1)
    assert repeated.metadata["reward_before_repetition_penalty"] == 0.0
    assert repeated.metadata["has_repetition"] is True
    assert normal.reward == 1.0
    assert normal.metadata["has_repetition"] is False


def test_penalizes_all_compaction_siblings_once() -> None:
    args = Namespace(repetition_reward_penalty=0.1, reward_key=None)
    repeated = _sample(rollout_id=7, reward=1.0, response="x" * 10_000, index=7)
    sibling = _sample(rollout_id=7, reward=1.0, response="short response", index=7)

    apply_repetition_reward_penalty(args, [repeated, sibling])
    apply_repetition_reward_penalty(args, [repeated, sibling])

    assert repeated.reward == pytest.approx(0.9)
    assert sibling.reward == pytest.approx(0.9)
    assert sibling.metadata["has_repetition"] is False
    assert sibling.metadata["repetition_reward_penalty_applied"] == 0.1


def test_zero_penalty_does_not_mutate_samples() -> None:
    args = Namespace(repetition_reward_penalty=0.0, reward_key=None)
    sample = _sample(rollout_id=1, reward=0.0, response="x" * 10_000, index=1)

    apply_repetition_reward_penalty(args, [sample])

    assert sample.reward == 0.0
    assert sample.metadata == {"raw_reward": 0.0}


def test_missing_penalty_argument_defaults_to_disabled() -> None:
    args = Namespace(reward_key=None)
    sample = _sample(rollout_id=1, reward=0.0, response="x" * 10_000, index=1)

    apply_repetition_reward_penalty(args, [sample])

    assert sample.reward == 0.0
    assert sample.metadata == {"raw_reward": 0.0}


def test_penalizes_selected_reward_key() -> None:
    args = Namespace(repetition_reward_penalty=0.1, reward_key="chess")
    sample = _sample(rollout_id=1, reward=0.0, response="x" * 10_000, index=1)
    sample.reward = {"chess": 1.0, "format": 0.5}
    sample.metadata["raw_reward"] = {"chess": 1.0, "format": 0.5}

    apply_repetition_reward_penalty(args, [sample])

    assert sample.reward == {"chess": pytest.approx(0.9), "format": 0.5}
    assert sample.metadata["raw_reward"] == {"chess": pytest.approx(0.9), "format": 0.5}


def test_rejects_negative_penalty() -> None:
    args = Namespace(repetition_reward_penalty=-0.1, reward_key=None)

    with pytest.raises(ValueError, match="must be nonnegative"):
        apply_repetition_reward_penalty(
            args,
            [_sample(rollout_id=1, reward=0.0, response="x" * 10_000, index=1)],
        )
