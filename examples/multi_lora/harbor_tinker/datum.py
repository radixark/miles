"""Trajectory JSON → Tinker Datums for the Harbor × Tinker loop.

Skeleton: functions document what they will do; bodies land in follow-up commits.
"""

from __future__ import annotations

from typing import Any

from miles.tinker.core.tinker_session_server import Turn
from miles.tinker.core.trajectory import Sequence


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Group-normalize one task's n rewards (subtract the mean, divide by the std; all zeros when the std is 0)."""
    raise NotImplementedError


def turns_from_trajectory(payload: dict[str, Any]) -> list[Turn]:
    """Rebuild Turn objects from the GET /oai/sessions/{sid} JSON."""
    raise NotImplementedError


def sequence_to_datum(sequence: Sequence, advantage: float):
    """Wrap trajectory.datum_arrays() into a tinker Datum: ModelInput.from_ints + TensorData loss inputs (target_tokens, logprobs, advantages)."""
    raise NotImplementedError


def trajectory_to_datums(payload: dict[str, Any], advantage: float) -> list:
    """turns_from_trajectory → to_sequences → one Datum per sequence (merged when the turns chain, else per turn)."""
    raise NotImplementedError
