# doc-dev: docs/ci/03-metric-history-gate.md
"""Offline regression gate for the CI metric-history system.

* Consumes one merged per-run JSONL record plus the `register_ci_gate` specs
  declared in the test file, and decides whether the run is *trusted*.
* The record is the passing attempt's merged JSONL, selected and merged by the
  harness caller (`tests.ci.ci_utils`) before the gate is invoked.
* Each spec pairs a STEPS selection (which value(s) to pull: `"last"` /
  `"all"` / a step list) with a CONSTRAINT (the pass/fail rule). `"all"` and a
  step list fan out to one comparison coordinate per step, each judged only
  against that same step's history.
* One check runs per coordinate, using the spec's constraint: the HISTORICAL
  gate. It is active only when the store returns >=1 trusted baseline value
  for the (identity, coordinate), and compares against the mean of those
  values. Zero trusted values means INACTIVE -- a cold start, not a failure.
* The run is trusted iff every *active* check passed for every coordinate.
* The gate is pure and read-only: the store is injected via
  :class:`MetricHistoryStore` and the only store call is
  `store.recent_trusted_values` -- no connection opened, no wandb read, no
  rows written.

Caveats:

* Do not add store writes here. Persistence is the caller's job --
  :func:`tests.ci.ci_utils.run_gate_hook` writes rows on nightly-marked runs --
  and the gate itself stays read-only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum

from tests.ci.ci_register import CIRegistry, HWBackend, ut_parse_one_file
from tests.ci.metric_history.constraints import evaluate_constraint
from tests.ci.metric_history.register import CiGateSpec, parse_ci_gate_specs
from tests.ci.metric_history.selection import SelectionError, select
from tests.ci.metric_history.storage import MetricHistoryStore

# Maps the parsed HWBackend enum to the lowercase backend string the store keys
# on. The store's identity tuple is all strings; CIRegistry.backend is the enum.
_BACKEND_STR: dict[HWBackend, str] = {
    HWBackend.CPU: "cpu",
    HWBackend.CUDA: "cuda",
    HWBackend.ROCM: "rocm",
}


class GateStatus(Enum):
    """Outcome of the historical check for one coordinate."""

    PASS = "pass"
    FAIL = "fail"
    INACTIVE = "inactive"  # check not applicable: historical cold start
    ERROR = "error"  # the metric could not be selected (missing/empty series, bad step)


@dataclass(frozen=True)
class MetricGateResult:
    """Per-coordinate verdict.

    `(metric_key, steps_key, constraint_key, step)` is the baseline coordinate;
    `step` is None only when selection errored (there is no coordinate to
    name), `-1` for a whole-series reduction like `steps="last"`. `at_step` is the
    step the value actually came from, for reporting. `current` is the
    extracted scalar, or None when selection errored. `baseline_mean` is the
    mean of trusted history when the historical gate is active, else None.
    `trusted` is True iff the check here passed or was inactive.
    """

    metric_key: str
    steps_key: str
    constraint_key: str
    step: int | None
    at_step: int | None
    current: float | None
    historical_status: GateStatus
    baseline_n: int
    baseline_mean: float | None
    reason: str

    @property
    def trusted(self) -> bool:
        return self.historical_status in (GateStatus.PASS, GateStatus.INACTIVE)


@dataclass(frozen=True)
class GateResult:
    """Run-level verdict over every gate spec for one test file."""

    test_path: str
    backend: str
    suite: str
    metrics: list[MetricGateResult] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """The run is trusted iff every per-coordinate verdict is trusted.

        An empty metrics list (no gate specs) is vacuously trusted: a file that
        declares no gate cannot regress. One failing step of a fanned-out spec
        untrusts the whole run.
        """
        return all(m.trusted for m in self.metrics)


# Capture serializes non-finite floats as these markers so record lines stay
# strict JSON; decode them back before extraction judges the values.
_NONFINITE_MARKERS: dict[str, float] = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def _decode_value(value: object) -> object:
    if isinstance(value, str) and value in _NONFINITE_MARKERS:
        return _NONFINITE_MARKERS[value]
    return value


def parse_merged_record(record_path: str) -> dict[str, list]:
    """Read a merged JSONL record into `{metric_key: series}`.

    Each line is `{"metric": key, "series": [[step, value], ...]}`. Non-finite
    values arrive as the capture-side string markers and are decoded back to
    floats here. A repeated metric key (should not happen post-merge) keeps the
    last line's series.
    """
    by_metric: dict[str, list] = {}
    with open(record_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_metric[rec["metric"]] = [
                [point[0], _decode_value(point[1])] if isinstance(point, list) and len(point) >= 2 else point
                for point in rec["series"]
            ]
    return by_metric


def _registry_for(filename: str) -> CIRegistry:
    """The single CIRegistry governing this test file.

    A gate needs exactly one (backend, suite) identity. A file with no register
    call, or more than one, is an authoring error the gate refuses rather than
    guessing which suite a metric belongs to.
    """
    registries = ut_parse_one_file(filename)
    if not registries:
        raise ValueError(f"{filename}: no register_*_ci() call; gate identity is undefined")
    if len(registries) > 1:
        raise ValueError(f"{filename}: {len(registries)} register_*_ci() calls; gate identity is ambiguous")
    return registries[0]


def _error_result(spec: CiGateSpec, reason: str) -> MetricGateResult:
    return MetricGateResult(
        metric_key=spec.metric_key,
        steps_key=spec.steps_key,
        constraint_key=spec.constraint_key,
        step=None,
        at_step=None,
        current=None,
        historical_status=GateStatus.ERROR,
        baseline_n=0,
        baseline_mean=None,
        reason=reason,
    )


def _evaluate_spec(
    spec: CiGateSpec,
    by_metric: dict[str, list],
    store: MetricHistoryStore,
    *,
    test_path: str,
    backend: str,
    suite: str,
    history_limit: int,
) -> list[MetricGateResult]:
    series = by_metric.get(spec.metric_key)
    if series is None:
        return [_error_result(spec, f"required metric {spec.metric_key!r} missing from record")]

    try:
        selections = select(series, spec.steps)
    except SelectionError as e:
        return [_error_result(spec, f"metric {spec.metric_key!r} (steps={spec.steps!r}): {e}")]

    results: list[MetricGateResult] = []
    for ex in selections:
        reasons: list[str] = []
        trusted_values = store.recent_trusted_values(
            test_path,
            backend,
            suite,
            spec.metric_key,
            spec.steps_key,
            spec.constraint_key,
            ex.step,
            history_limit,
        )
        if not trusted_values:
            historical_status = GateStatus.INACTIVE
            baseline_mean = None
            reasons.append("historical: cold start (0 trusted baselines)")
        else:
            baseline_mean = sum(trusted_values) / len(trusted_values)
            hist = evaluate_constraint(spec.constraint, ex.value, baseline_mean)
            historical_status = GateStatus.PASS if hist.ok else GateStatus.FAIL
            if not hist.ok:
                reasons.append(
                    f"historical: cur={ex.value:.6g} vs mean={baseline_mean:.6g} "
                    f"(n={len(trusted_values)}) outside [{hist.lo:.6g}, {hist.hi:.6g}]"
                )

        if historical_status in (GateStatus.PASS, GateStatus.INACTIVE):
            reasons.insert(0, "ok")

        results.append(
            MetricGateResult(
                metric_key=spec.metric_key,
                steps_key=spec.steps_key,
                constraint_key=spec.constraint_key,
                step=ex.step,
                at_step=ex.at_step,
                current=ex.value,
                historical_status=historical_status,
                baseline_n=len(trusted_values),
                baseline_mean=baseline_mean,
                reason="; ".join(reasons),
            )
        )
    return results


def evaluate_gate(
    test_filename: str,
    merged_record_path: str,
    store: MetricHistoryStore,
    *,
    registry: CIRegistry | None = None,
    history_limit: int = 20,
) -> GateResult:
    """Evaluate every `register_ci_gate` spec in `test_filename` against a record.

    `test_filename` is the repo-relative test path. Gate identity
    (test_path/backend/suite) comes from `registry` when the real harness
    passes one -- it has already chosen which `register_*_ci()` call applies,
    so a file with several (e.g. `register_cuda_ci` + `register_rocm_ci`)
    is handled without reparsing. When `registry` is None and the file has
    gate specs, identity is reparsed via `_registry_for` (the isolated
    unit-test convenience, which still refuses a no-register or ambiguous file).

    A file with no gate specs is vacuously trusted and does NOT require a unique
    registry: identity is taken from `registry` if given, else filled
    best-effort from `test_filename` without raising on a dual-register or
    no-register file.

    `merged_record_path` is the merged per-run JSONL of the passed attempt --
    the gate never globs a base directory to find it. `store` answers the
    baseline query and nothing else (no writes, no connection opened here). A
    fanned-out spec contributes one MetricGateResult per step.
    """
    specs = parse_ci_gate_specs(test_filename)
    if not specs:
        # No gate spec can regress, so identity is informational here and must
        # never raise on a dual-register / no-register file. Use the harness's
        # registry if it gave one, otherwise fill best-effort from the filename.
        if registry is not None:
            return GateResult(
                test_path=registry.filename,
                backend=_BACKEND_STR[registry.backend],
                suite=registry.suite,
                metrics=[],
            )
        return GateResult(
            test_path=test_filename,
            backend="",
            suite="",
            metrics=[],
        )

    # The harness already selected one register_*_ci() call; use it directly and
    # do not reparse (the file may carry several register calls). With no registry
    # this is the isolated-unit-test path, which reparses and may still raise on
    # an ambiguous (multi-register) or no-register file.
    if registry is None:
        registry = _registry_for(test_filename)
    backend = _BACKEND_STR[registry.backend]
    by_metric = parse_merged_record(merged_record_path)

    results: list[MetricGateResult] = []
    for spec in specs:
        results.extend(
            _evaluate_spec(
                spec,
                by_metric,
                store,
                test_path=registry.filename,
                backend=backend,
                suite=registry.suite,
                history_limit=history_limit,
            )
        )

    return GateResult(
        test_path=registry.filename,
        backend=backend,
        suite=registry.suite,
        metrics=results,
    )
