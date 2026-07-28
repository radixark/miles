# doc-dev: docs/ci/03-metric-history-gate.md
"""Step selection for the CI regression gate.

* `select(series, steps)` pulls the comparison value(s) out of one metric's
  per-run series `[[step, value], ...]`; step may be None.
* `steps` is the declaration's flat literal, validated by the parser:
  `"last"` | `"all"` | a non-empty list of step indices. There is no
  name-keyed registry -- the selection space is a closed enum.
* Selection is pure and returns a list of :class:`Selection` -- one entry per
  comparison coordinate.
* `"last"` -- the last numeric point (1 coordinate, `step = -1`).
* `"all"` -- every step present in the series, fanned out (N coordinates).
* `[k, ...]` -- the named steps, fanned out (`len(steps)` coordinates); a named
  step missing from the series is an error, not a silent skip.
* A fanned coordinate is identified by its step, so this run's step-0 value is
  compared only against past runs' step-0 values.
* Raising :class:`SelectionError` (rather than returning a sentinel) lets the
  gate turn the failure into a clear per-coordinate verdict.

Caveats:

* A non-finite value (NaN/±Inf) at a coordinate the selection picks is an
  SelectionError -- judged, never silently dropped. Points whose value is not a
  number at all (bool/None/...) are ignored, as are points no selection picks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point = Sequence  # [step, value]


class SelectionError(ValueError):
    """A required series is absent, empty, or ill-formed for this selection."""


@dataclass(frozen=True)
class Selection:
    """One comparison value pulled from a series.

    `step` is the identity role: step `k` for a per-step value, `-1` for a
    whole-series reduction (`"last"`), which must key on a constant or its
    history would fragment across runs of different lengths. `at_step` is the
    step the value actually came from, carried for reporting only (None when
    the series carries no step index).
    """

    step: int
    at_step: int | None
    value: float


def _is_number(value: object) -> bool:
    """A real int/float (not bool — it sneaks through `isinstance(x, int)`).

    Finiteness is deliberately not checked here: non-finite values stay in the
    series and error at selection time, never silently dropped.
    """
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _valid_step(step: object) -> bool:
    return isinstance(step, int) and not isinstance(step, bool)


def _numeric_points(series: Sequence[Point]) -> list[tuple[int | None, float]]:
    """(step, value) for each point whose value is a number.

    A non-int (or bool) step is normalized to None; the value is kept —
    including non-finite floats, which selection rejects if picked. Points
    with a non-numeric value are dropped.
    """
    out: list[tuple[int | None, float]] = []
    for point in series:
        if len(point) < 2:
            continue
        step, value = point[0], point[1]
        if not _is_number(value):
            continue
        out.append((step if _valid_step(step) else None, float(value)))
    return out


def _select_last(series: Sequence[Point]) -> list[Selection]:
    points = _numeric_points(series)
    if not points:
        raise SelectionError("series has no numeric point")
    step, value = points[-1]
    if not math.isfinite(value):
        raise SelectionError(f"last: non-finite value {value!r} at the last point (step {step})")
    return [Selection(step=-1, at_step=step, value=value)]


def _select_all(series: Sequence[Point]) -> list[Selection]:
    points = _numeric_points(series)
    if not points:
        raise SelectionError("series has no numeric point")
    out: list[Selection] = []
    seen: set[int] = set()
    for step, value in points:
        if step is None:
            raise SelectionError("all: a numeric point carries no step index")
        if step in seen:
            raise SelectionError(f"all: duplicate step {step} in series")
        if not math.isfinite(value):
            raise SelectionError(f"all: non-finite value {value!r} at step {step}")
        seen.add(step)
        out.append(Selection(step=step, at_step=step, value=value))
    return out


def _select_steps(series: Sequence[Point], steps: Sequence[int]) -> list[Selection]:
    by_step: dict[int, float] = {}
    for step, value in _numeric_points(series):
        if step is None:
            continue
        if step in by_step:
            raise SelectionError(f"steps: duplicate step {step} in series")
        by_step[step] = value
    out: list[Selection] = []
    for k in steps:
        if k not in by_step:
            raise SelectionError(f"steps: required step {k} missing from series")
        if not math.isfinite(by_step[k]):
            raise SelectionError(f"steps: non-finite value {by_step[k]!r} at required step {k}")
        out.append(Selection(step=k, at_step=k, value=by_step[k]))
    return out


def select(series: Sequence[Point], steps: str | Sequence[int]) -> list[Selection]:
    """Apply a validated `steps` literal to a series."""
    if steps == "last":
        return _select_last(series)
    if steps == "all":
        return _select_all(series)
    if isinstance(steps, (list, tuple)):
        return _select_steps(series, steps)
    raise SelectionError(f'invalid steps {steps!r}; valid: "last", "all", or a list of step indices')
