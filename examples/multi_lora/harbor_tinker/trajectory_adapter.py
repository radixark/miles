"""Collector turns → tinker-cookbook Trajectory.

Skeleton: functions document what they will do; bodies land in follow-up commits.

This is the only data-shaping code of our own. A recorded ``Turn`` is a cookbook ``Transition`` (``ob`` = the
prompt ids the engine saw, ``ac`` = the sampled ids and their logprobs); everything downstream — merging or
splitting turns into Datums (``trajectory_to_data``), group advantages (``compute_advantages``), batching and
the ``mask`` handling (``_remove_mask``) — is ``tinker_cookbook.rl`` unchanged.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.rl.types import Trajectory, Transition


def turn_to_transition(turn: dict[str, Any], *, episode_done: bool) -> Transition:
    """One collector turn → Transition(ob=ModelInput.from_ints(input_ids), ac=TokensWithLogprobs(output_ids, logprobs, finish_reason), reward=0.0, episode_done)."""
    raise NotImplementedError


def turns_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    """GET /oai/sessions/{sid} JSON → Trajectory(transitions, final_ob = last input_ids + output_ids, stop_reason = last finish_reason)."""
    raise NotImplementedError
