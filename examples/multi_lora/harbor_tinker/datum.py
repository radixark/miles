"""Collector turns → tinker-cookbook Trajectory → tinker Datums.

Skeleton: functions document what they will do; bodies land in follow-up commits.

No merging logic of our own: a recorded ``Turn`` is a cookbook ``Transition`` (``ob`` = the prompt ids the
engine saw, ``ac`` = the sampled ids and their logprobs), so a session's turns become a ``Trajectory`` and
``tinker_cookbook.rl.data_processing.trajectory_to_data`` decides whether they chain into one Datum or split
per turn, exactly as it does for cookbook RL envs. Group advantages come from ``compute_advantages``.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

import tinker


def turn_to_transition(turn: dict[str, Any], *, episode_done: bool) -> Transition:
    """One collector turn → Transition(ob=ModelInput.from_ints(input_ids), ac=TokensWithLogprobs(output_ids, logprobs, finish_reason), reward=0.0)."""
    raise NotImplementedError


def turns_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    """GET /oai/sessions/{sid} JSON → Trajectory(transitions, final_ob = last input_ids + output_ids, stop_reason = last finish_reason)."""
    raise NotImplementedError


def build_trajectory_group(trajectories: list[Trajectory], rewards: list[float]) -> TrajectoryGroup:
    """One task's n trajectories with their Harbor rewards as final_rewards_G (transition rewards stay 0)."""
    raise NotImplementedError


def group_to_datums(group: TrajectoryGroup) -> list[tinker.Datum]:
    """compute_advantages([group]) then trajectory_to_data(trajectory, advantage) for each trajectory; flatten."""
    raise NotImplementedError


def strip_mask(datum: tinker.Datum) -> tinker.Datum:
    """Stopgap until the gateway accepts Tinker's optional `mask` loss input: fold mask into advantages and drop the key."""
    raise NotImplementedError
