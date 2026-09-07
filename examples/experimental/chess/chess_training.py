"""Chess reward policy and trajectory metrics using existing Miles hooks."""

from argparse import Namespace
from collections.abc import Mapping
from copy import copy
from math import isfinite
from typing import Any

from miles.ray.rollout.metrics import log_rollout_data
from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.utils.metric_utils import has_repetition
from miles.utils.types import Sample


def _invalid_move_termination(metadata: Mapping[str, Any]) -> bool:
    result = metadata.get("chess_result")
    flag = result.get("invalid_move_termination") if isinstance(result, Mapping) else None
    if type(flag) is not bool:
        raise ValueError("Chess harness must report a boolean invalid_move_termination")
    return flag


def postprocess_samples(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Zero invalid-move base rewards, then penalize every repetitive game."""
    agent = session_metadata["agent"]
    invalid = _invalid_move_termination(agent)
    penalty = agent["repetition_reward_penalty"]
    if type(penalty) not in (int, float) or not isfinite(penalty) or penalty < 0:
        raise ValueError("repetition_reward_penalty must be finite and non-negative")

    samples = default_postprocess(leaf_samples, session_metadata)
    flags = [has_repetition(sample.response) for sample in samples]
    applied_penalty = penalty if any(flags) else 0.0
    for sample, repeated in zip(samples, flags, strict=True):
        base_reward = 0.0 if invalid else sample.reward
        sample.reward = base_reward - applied_penalty
        sample.metadata.update(
            has_repetition=repeated,
            reward_before_repetition_penalty=base_reward,
            repetition_reward_penalty_applied=applied_penalty,
            raw_reward=sample.reward,
        )
    return samples


def log_rollout_metrics(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Extend the standard logger, counting each original game exactly once."""
    by_game = {}
    for position, sample in enumerate(samples):
        invalid = _invalid_move_termination(sample.metadata)
        if sample.rollout_id is not None:
            identity = ("rollout", sample.rollout_id)
        elif sample.index is not None:
            identity = ("sample", sample.index)
        else:
            identity = ("position", position)
        key = (sample.adapter, sample.group_index, identity)
        if key in by_game and by_game[key] != invalid:
            raise ValueError("Sibling samples disagree on invalid-move termination")
        by_game[key] = invalid
    count = sum(by_game.values())
    metrics = dict(rollout_extra_metrics or {})
    metrics.update(
        {
            "rollout/chess/invalid_move_termination_count": count,
            "rollout/chess/trajectory_count": len(by_game),
            "rollout/chess/invalid_move_termination_rate": count / len(by_game) if by_game else 0.0,
        }
    )
    # Reuse the standard logger without recursively invoking this hook.
    default_args = copy(args)
    default_args.custom_rollout_log_function_path = None
    log_rollout_data(rollout_id, default_args, samples, metrics, rollout_time)
    return True
