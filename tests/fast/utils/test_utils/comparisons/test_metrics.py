from datetime import datetime, timezone
from typing import Any

from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.test_utils.comparisons.metrics import _check_single_metric, _keep_only_final_attempt

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = MainProcessIdentity()


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
