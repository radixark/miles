# doc-dev: docs/developer/ci/03-metric-history-gate.md
"""The band constraint for the CI regression gate.

* A constraint decides whether one extracted scalar `cur` passes against a
  reference `ref` -- the mean of the trusted baseline for the historical
  gate.
* Every constraint has a lower bound `ref - band_down`. An upper bound
  `ref + band_up` applies only when an upper band param is declared; omitting
  both upper params leaves improvements unbounded, as for raw reward.
* Each side's band scales from the reference only:
  `band = max(rel * |ref|, abs_floor)`, so `rel_up = 0.5` reads as "at most
  50% above the baseline mean". Scaling from `cur` would let a deviating
  value widen its own tolerance (at rel 0.5 the effective ceiling is 2x ref,
  not 1.5x). `abs_floor` keeps metrics near zero from flagging on a
  meaningless relative percentage.
* The declaration literal is validated against :data:`CONSTRAINT_SCHEMA` at
  parse time; `evaluate_constraint` expects the normalized dict (defaults
  filled for declared sides). The literal as written is the spec's
  `constraint_key`, stored with each value as provenance of the rule that
  judged it (see register.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintOutcome:
    """Whether `cur` passed, and the corridor `[lo, hi]` it was judged against."""

    ok: bool
    lo: float
    hi: float


# Parse-time param schema, consumed by register.py. Each entry:
# param -> (validator_key, required, default). A declaration must write at
# least one lower band param. The upper side is optional; defaults are filled
# only for declared sides, so an explicit zero band still pins that bound to the reference.
CONSTRAINT_SCHEMA: dict[str, tuple[str, bool, object]] = {
    "rel_up": ("float_nonneg", False, 0.0),
    "abs_floor_up": ("float_nonneg", False, 0.0),
    "rel_down": ("float_nonneg", False, 0.0),
    "abs_floor_down": ("float_nonneg", False, 0.0),
}


def evaluate_constraint(constraint: dict, cur: float, ref: float) -> ConstraintOutcome:
    """Apply a normalized constraint dict to `cur` vs `ref`."""
    hi = ref + max(constraint["rel_up"] * abs(ref), constraint["abs_floor_up"]) if "rel_up" in constraint else math.inf
    lo = ref - max(constraint["rel_down"] * abs(ref), constraint["abs_floor_down"])
    return ConstraintOutcome(lo <= cur <= hi, lo, hi)
