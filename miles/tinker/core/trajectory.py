"""Recorded turns → training sequences, SDK-free.

Skeleton: functions document what they will do; bodies land in follow-up commits.

Two shapes come out of one trajectory. When every turn's ``input_ids`` extends the previous turn's
``input_ids + output_ids`` the turns chain into one contiguous sequence (loss on the outputs, zero on
prompt and gaps). Otherwise each turn is its own sequence, which is exact on-policy by construction
because the turn's ``input_ids`` is what the engine actually saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miles.tinker.core.tinker_session_server import Turn


@dataclass
class Sequence:
    """One trainable sequence: tokens, a loss mask over the same positions, and sampling logprobs (0.0 where the mask is 0)."""

    tokens: list[int]
    loss_mask: list[int]
    logprobs: list[float]


def chain(turns: list[Turn]) -> Sequence | None:
    """Merge the turns into one sequence when each input_ids extends the previous input_ids + output_ids; otherwise None."""
    raise NotImplementedError


def per_turn(turns: list[Turn]) -> list[Sequence]:
    """One sequence per turn: input_ids + output_ids with loss only on the output ids."""
    raise NotImplementedError


def to_sequences(turns: list[Turn]) -> list[Sequence]:
    """chain() when every turn chains, otherwise per_turn()."""
    raise NotImplementedError


def datum_arrays(sequence: Sequence, advantage: float) -> dict[str, Any]:
    """Shape Tinker loss inputs: model_input = tokens[:-1], target_tokens = tokens[1:], logprobs/advantages zero before the first loss position, all of length len(tokens) - 1."""
    raise NotImplementedError
