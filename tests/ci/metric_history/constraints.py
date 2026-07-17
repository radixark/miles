# doc-dev: docs/ci/03-metric-history-gate.md
"""The band constraint for the CI regression gate.

* A constraint decides whether one extracted scalar `cur` passes against a
  reference `ref` -- the mean of the trusted baseline for the historical
  gate.
* One symmetric band family, no name dispatch:
  `band = max(rel * |cur|, rel * |ref|, abs_floor)`. `rel` is a relative
  percentage; scaling from both magnitudes makes operand order irrelevant,
  while `abs_floor` keeps metrics near zero from flagging on a meaningless
  relative percentage.
* `direction` narrows what counts as a failure: `two_sided` -- any
  deviation beyond the band fails; `higher_is_worse` -- only an increase
  beyond the band fails (a drop passes); `lower_is_worse` -- only a decrease
  beyond the band fails (a rise passes).
* The declaration literal is validated against :data:`CONSTRAINT_SCHEMA` at
  parse time; `evaluate_constraint` expects the normalized dict (defaults
  filled). The literal as written is the spec's `constraint_key`, part of the
  stored value's identity (see register.py).
"""

from __future__ import annotations

from dataclasses import dataclass

DIRECTIONS = ("two_sided", "higher_is_worse", "lower_is_worse")


@dataclass(frozen=True)
class ConstraintOutcome:
    """Whether `cur` passed, and the tolerance band that was applied."""

    ok: bool
    band: float


def _within(cur: float, ref: float, band: float, direction: str) -> bool:
    if direction == "higher_is_worse":
        return (cur - ref) <= band
    if direction == "lower_is_worse":
        return (ref - cur) <= band
    return abs(cur - ref) <= band


# Parse-time param schema, consumed by register.py. Each entry:
# param -> (validator_key, required, default). A declaration must write at
# least one band param (`rel` / `abs_floor`) -- the parser enforces it, since
# an all-default band of 0 fails on any deviation.
CONSTRAINT_SCHEMA: dict[str, tuple[str, bool, object]] = {
    "rel": ("float_nonneg", False, 0.0),
    "abs_floor": ("float_nonneg", False, 0.0),
    "direction": ("direction", False, "two_sided"),
}


def evaluate_constraint(constraint: dict, cur: float, ref: float) -> ConstraintOutcome:
    """Apply a normalized constraint dict to `cur` vs `ref`."""
    band = max(constraint["rel"] * abs(cur), constraint["rel"] * abs(ref), constraint["abs_floor"])
    return ConstraintOutcome(_within(cur, ref, band, constraint["direction"]), band)
