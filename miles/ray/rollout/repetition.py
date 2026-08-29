"""Repetition-aware reward shaping for rollout samples."""

from collections import defaultdict
from numbers import Real
from typing import Any

from miles.utils.metric_utils import has_repetition
from miles.utils.types import AdapterRef, Sample


_RolloutKey = tuple[AdapterRef | None, str, int | None, int]


def _rollout_key(sample: Sample, position: int) -> _RolloutKey:
    if sample.rollout_id is not None:
        return (sample.adapter, "rollout", sample.group_index, sample.rollout_id)
    if sample.index is not None:
        return (sample.adapter, "sample", sample.group_index, sample.index)
    return (sample.adapter, "position", sample.group_index, position)


def _set_reward_value(sample: Sample, args: Any, value: float) -> None:
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        if not isinstance(sample.reward, dict):
            raise TypeError("reward_key requires sample.reward to be a mapping")
        sample.reward = {**sample.reward, reward_key: value}
    else:
        sample.reward = value


def _set_metadata_raw_reward(sample: Sample, args: Any, value: float) -> None:
    raw_reward = sample.metadata.get("raw_reward")
    if raw_reward is None:
        return

    sample.metadata.setdefault("raw_reward_before_repetition_penalty", raw_reward)
    reward_key = getattr(args, "reward_key", None)
    if reward_key and isinstance(raw_reward, dict):
        sample.metadata["raw_reward"] = {**raw_reward, reward_key: value}
    else:
        sample.metadata["raw_reward"] = value


def apply_repetition_reward_penalty(args: Any, samples: list[Sample]) -> None:
    """Subtract the configured penalty once from each repetitive rollout.

    TITO compaction can split one rollout into several training samples. If any
    sibling sample contains repetition, every sibling receives the same adjusted
    reward so rollout-level advantage normalization remains well-defined.
    """
    penalty = getattr(args, "repetition_reward_penalty", 0.0)
    if penalty == 0 or not samples:
        return
    if penalty < 0:
        raise ValueError("repetition_reward_penalty must be nonnegative")

    sample_indices_by_rollout: dict[_RolloutKey, list[int]] = defaultdict(list)
    repetition_flags = [has_repetition(sample.response) for sample in samples]
    for position, sample in enumerate(samples):
        sample.metadata["has_repetition"] = repetition_flags[position]
        sample_indices_by_rollout[_rollout_key(sample, position)].append(position)

    for sample_indices in sample_indices_by_rollout.values():
        if not any(repetition_flags[index] for index in sample_indices):
            continue

        for index in sample_indices:
            sample = samples[index]
            if sample.metadata.get("repetition_reward_penalty_applied") == penalty:
                continue
            reward = sample.get_reward_value(args)
            if not isinstance(reward, Real):
                raise TypeError(f"repetition penalty requires a numeric reward, got {reward!r}")
            adjusted_reward = float(reward) - penalty
            sample.metadata.setdefault("reward_before_repetition_penalty", float(reward))
            sample.metadata["repetition_reward_penalty_applied"] = penalty
            _set_metadata_raw_reward(sample, args, adjusted_reward)
            _set_reward_value(sample, args, adjusted_reward)
