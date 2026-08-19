from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.comparisons.metrics import (
    _check_events_line_up,
    _check_single_metric,
    _keep_only_final_attempt,
    assert_metric_finite_and_nonzero,
)

_KEY: str = "train/grad_norm"

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = SimpleProcessIdentity(component="main")


def _metric_event(
    *, rollout_id: int | None, attempt: int | None, metrics: dict[str, Any] | None = None
) -> MetricEvent:
    return MetricEvent(
        timestamp=_FIXED_TS,
        source=_FIXED_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        metrics=metrics if metrics is not None else {},
    )


class TestKeepOnlyFinalAttempt:
    def test_keeps_highest_attempt_for_single_rollout(self) -> None:
        """Among attempts 0,1,2 for one rollout_id, only the attempt=2 event survives."""
        events = [
            _metric_event(rollout_id=1, attempt=0),
            _metric_event(rollout_id=1, attempt=1),
            _metric_event(rollout_id=1, attempt=2),
        ]
        kept = _keep_only_final_attempt(events)
        assert [e.attempt for e in kept] == [2]

    def test_highest_attempt_resolved_independently_per_rollout(self) -> None:
        """Each rollout_id keeps its own max attempt; different maxima coexist."""
        events = [
            _metric_event(rollout_id=1, attempt=0),
            _metric_event(rollout_id=1, attempt=1),
            _metric_event(rollout_id=2, attempt=0),
        ]
        kept = _keep_only_final_attempt(events)
        assert {(e.rollout_id, e.attempt) for e in kept} == {(1, 1), (2, 0)}

    def test_none_attempt_normalized_to_zero_and_dropped_when_mixed(self) -> None:
        """attempt=None normalizes to 0, so it is dropped when an attempt=1 event shares the rollout_id."""
        events = [
            _metric_event(rollout_id=1, attempt=None),
            _metric_event(rollout_id=1, attempt=1),
        ]
        kept = _keep_only_final_attempt(events)
        assert [e.attempt for e in kept] == [1]

    def test_empty_input_returns_empty(self) -> None:
        """An empty event list yields an empty result."""
        assert _keep_only_final_attempt([]) == []

    def test_ties_on_max_attempt_all_kept(self) -> None:
        """Multiple events tied at the max attempt for a rollout_id are all retained."""
        events = [
            _metric_event(rollout_id=1, attempt=2, metrics={"a": 1}),
            _metric_event(rollout_id=1, attempt=2, metrics={"b": 2}),
        ]
        kept = _keep_only_final_attempt(events)
        assert len(kept) == 2
        assert [e.metrics for e in kept] == [{"a": 1}, {"b": 2}]


class TestCheckSingleMetric:
    def test_equal_values_no_issue(self) -> None:
        """Exactly equal numeric values produce no issue."""
        assert _check_single_metric(0, "k", 1.5, 1.5, rtol=0.01, atol=0.0) == []

    def test_within_atol_no_issue(self) -> None:
        """A difference within atol is accepted even if relative difference would exceed rtol."""
        assert _check_single_metric(0, "k", 1.0, 1.0 + 1e-9, rtol=0.0, atol=1e-6) == []

    def test_relative_difference_above_rtol_reports_issue(self) -> None:
        """A relative difference above rtol (and above atol) yields exactly one issue."""
        issues = _check_single_metric(3, "train/loss", 1.0, 2.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "train/loss" in issues[0]
        assert "rel_diff" in issues[0]

    def test_nan_detected(self) -> None:
        """A NaN on either side produces a 'NaN detected' issue."""
        issues = _check_single_metric(0, "k", float("nan"), 1.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "NaN detected" in issues[0]

    def test_matching_inf_no_issue(self) -> None:
        """inf == inf compares equal and produces no issue."""
        assert _check_single_metric(0, "k", float("inf"), float("inf"), rtol=0.1, atol=0.0) == []

    def test_inf_vs_finite_reports_mismatch(self) -> None:
        """inf versus a finite value produces an 'inf mismatch' issue."""
        issues = _check_single_metric(0, "k", float("inf"), 1.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "inf mismatch" in issues[0]

    def test_both_zero_no_issue(self) -> None:
        """Two exact zeros short-circuit to no issue."""
        assert _check_single_metric(0, "k", 0.0, 0.0, rtol=0.0, atol=0.0) == []

    def test_non_numeric_skipped(self) -> None:
        """A non-numeric value on either side is skipped (no issue)."""
        assert _check_single_metric(0, "k", "abc", 1.0, rtol=0.0, atol=0.0) == []

    def test_tiny_baseline_uses_relative_floor(self) -> None:
        """A near-zero baseline uses the 1e-12 denominator floor, making a tiny abs diff a large rel diff."""
        issues = _check_single_metric(0, "k", 0.0, 5e-13, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "rel_diff" in issues[0]


class TestCheckEventsLineUp:
    def test_two_sides_describing_different_rollouts_are_reported(self) -> None:
        """Comparing by read order passes silently when the sides are offset but the numbers happen to agree."""
        baseline = [_metric_event(rollout_id=0, attempt=0), _metric_event(rollout_id=1, attempt=0)]
        target = [_metric_event(rollout_id=1, attempt=0), _metric_event(rollout_id=2, attempt=0)]

        issues = _check_events_line_up(baseline, target)

        assert len(issues) == 2

    def test_a_retried_rollout_still_lines_up_with_its_baseline(self) -> None:
        """The target retries a crashed rollout, and comparing its winning attempt is the point of the run."""
        baseline = [_metric_event(rollout_id=3, attempt=0)]
        target = [_metric_event(rollout_id=3, attempt=1)]

        assert _check_events_line_up(baseline, target) == []


class TestAssertMetricWasFiniteAndNonzero:
    def test_a_run_that_trained_as_many_rollouts_as_asked_passes(self, dump_dir) -> None:
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_metrics(dump_dir, [(0, 0.5), (1, 0.4)])

        assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)

    def test_one_rollout_reported_over_several_steps_is_not_several_rollouts(self, dump_dir) -> None:
        """Several optimizer steps of one rollout say that one rollout trained, however many events they write."""
        _write_metrics(dump_dir, [(0, 0.5), (0, 0.4), (0, 0.3)])

        with pytest.raises(AssertionError, match="in only 1 of 1 rollout"):
            assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)

    def test_a_rollout_whose_only_usable_step_is_zero_does_not_count(self, dump_dir) -> None:
        """A gradient of zero moved no weights, which is what this assertion exists to catch."""
        _write_metrics(dump_dir, [(0, 0.5), (1, 0.0)])

        with pytest.raises(AssertionError, match="in only 1 of 2 rollout"):
            assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)


@pytest.fixture
def dump_dir(tmp_path) -> str:
    return str(tmp_path / "run")


def _write_metrics(dump_dir: str, points: list[tuple[int, float]]) -> None:
    event_logger = EventLogger(log_dir=Path(dump_dir) / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
    for rollout_id, value in points:
        event_logger.log(MetricEvent, dict(rollout_id=rollout_id, attempt=0, metrics={_KEY: value}), print_log=False)
    event_logger.close()
