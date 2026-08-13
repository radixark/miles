from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent, WitnessAllocateIdEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.comparisons.metrics import (
    _check_events_line_up,
    _check_single_metric,
    _check_step_metrics,
    _keep_only_final_attempt,
    assert_gradients_nonzero,
    assert_metrics_classified,
    read_metric_series,
    read_rollout_completion_times,
    assert_metrics_classified,
    read_metric_events,
    read_metric_series,
    read_rollout_completion_times,
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


class TestReadMetricEvents:
    def test_only_metric_events_are_returned_in_emission_order(self, tmp_path: Path) -> None:
        """Mixed event streams yield only metric events in their original emission order."""
        event_logger = EventLogger(log_dir=tmp_path, source=_FIXED_SOURCE, file_name="main.jsonl")
        event_logger.log(MetricEvent, dict(metrics={"sequence": 1}), print_log=False)
        event_logger.log(
            WitnessAllocateIdEvent,
            dict(rollout_id=0, attempt=0, witness_id_to_sample_index={10: 0}, counter_after=11),
            print_log=False,
        )
        event_logger.log(MetricEvent, dict(metrics={"sequence": 2}), print_log=False)
        event_logger.close()

        events = read_metric_events(tmp_path)

        assert all(isinstance(event, MetricEvent) for event in events)
        assert [event.metrics for event in events] == [{"sequence": 1}, {"sequence": 2}]

    def test_absent_events_directory_returns_empty(self, tmp_path: Path) -> None:
        """An events directory that does not exist yields an empty metric stream."""
        assert read_metric_events(tmp_path / "absent") == []


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


class TestCheckStepMetrics:
    def test_matching_selected_keys_report_nothing(self) -> None:
        """Both sides carrying the same selected keys with the same values is the passing case."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_a_key_only_the_baseline_has_is_reported(self) -> None:
        """A metric the target stopped producing must not vanish from the comparison."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in baseline but missing in target" in issues[0]

    def test_a_key_only_the_target_has_is_reported(self) -> None:
        """Regression: a metric the target alone produces used to pass unnoticed."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in target but missing in baseline" in issues[0]

    def test_a_target_only_key_outside_the_prefixes_is_ignored(self) -> None:
        """Only the namespaces this comparison claims to cover are compared."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"perf/step_time": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_an_excluded_target_only_key_is_ignored(self) -> None:
        """An explicitly excluded key stays excluded whichever side carries it."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0, exclude_keys=["train/loss"])

        assert issues == []

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


class TestCheckStepMetrics:
    def test_matching_selected_keys_report_nothing(self) -> None:
        """Both sides carrying the same selected keys with the same values is the passing case."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_a_key_only_the_baseline_has_is_reported(self) -> None:
        """A metric the target stopped producing must not vanish from the comparison."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in baseline but missing in target" in issues[0]

    def test_a_key_only_the_target_has_is_reported(self) -> None:
        """Regression: a metric the target alone produces used to pass unnoticed."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in target but missing in baseline" in issues[0]

    def test_a_target_only_key_outside_the_prefixes_is_ignored(self) -> None:
        """Only the namespaces this comparison claims to cover are compared."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"perf/step_time": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_an_excluded_target_only_key_is_ignored(self) -> None:
        """An explicitly excluded key stays excluded whichever side carries it."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0, exclude_keys=["train/loss"])

        assert issues == []


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


class TestReadRolloutCompletionTimes:
    def test_only_rollout_executor_events_are_completion_times(self, dump_dir: str) -> None:
        """Trainer metrics sharing a rollout id must not be mistaken for rollout completion."""
        events_dir = Path(dump_dir) / "events"
        rollout_logger = EventLogger(
            log_dir=events_dir,
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="rollout.jsonl",
        )
        trainer_logger = EventLogger(log_dir=events_dir, source=_FIXED_SOURCE, file_name="trainer.jsonl")
        rollout_logger.log(MetricEvent, dict(rollout_id=3, attempt=0, metrics={}), print_log=False)
        trainer_logger.log(MetricEvent, dict(rollout_id=4, attempt=0, metrics={}), print_log=False)
        rollout_logger.close()
        trainer_logger.close()

        completion_times = read_rollout_completion_times(dump_dir)

        assert [rollout_id for rollout_id, _timestamp in completion_times] == [3]

    def test_identified_rollout_executor_events_are_sorted_by_completion_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Completion times contain every identified executor event ordered by its timestamp."""
        timestamps = iter(
            [
                datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            ]
        )

        class _SequencedDatetime(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                return next(timestamps)

        monkeypatch.setattr("miles.utils.audit_utils.event_logger.logger.datetime", _SequencedDatetime)
        dump_dir = tmp_path / "run"
        event_logger = EventLogger(
            log_dir=dump_dir / "events",
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="main.jsonl",
        )
        for rollout_id, attempt in [(20, 0), (None, 0), (10, 0), (20, 1)]:
            event_logger.log(
                MetricEvent,
                {"rollout_id": rollout_id, "attempt": attempt, "metrics": {}},
                print_log=False,
            )
        event_logger.close()

        assert read_rollout_completion_times(str(dump_dir)) == [
            (20, datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
            (10, datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)),
            (20, datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc)),
        ]

    def test_final_identified_rollouts_are_sorted_by_completion_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Identified executor completions are returned in timestamp order, not emission order."""
        timestamps = iter(
            [
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ]
        )

        class _SequencedDatetime(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                return next(timestamps)

        monkeypatch.setattr("miles.utils.audit_utils.event_logger.logger.datetime", _SequencedDatetime)
        dump_dir = tmp_path / "run"
        event_logger = EventLogger(
            log_dir=dump_dir / "events",
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="main.jsonl",
        )
        for rollout_id in (2, 1):
            event_logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {}}, print_log=False)
        event_logger.close()

        assert [rollout_id for rollout_id, _ in read_rollout_completion_times(str(dump_dir))] == [1, 2]


class TestReadMetricSeries:
    def test_only_numeric_values_from_the_final_attempt_are_returned(self, dump_dir: str) -> None:
        """Only numeric final-attempt values survive, with integers normalized to floats."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={_KEY: 99.0}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 1}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 2.5}),
                _metric_event(rollout_id=1, attempt=0, metrics={_KEY: True}),
                _metric_event(rollout_id=2, attempt=0, metrics={"train/loss": 3.0}),
                _metric_event(rollout_id=3, attempt=0, metrics={_KEY: "4.0"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(0, 1.0), (0, 2.5)]

    def test_only_numeric_values_from_final_attempts_form_the_series(self, dump_dir: str) -> None:
        """Superseded, Boolean, missing, and textual metric values do not enter the final series."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=4, attempt=0, metrics={_KEY: 9.0}),
                _metric_event(rollout_id=4, attempt=1, metrics={_KEY: 3}),
                _metric_event(rollout_id=5, attempt=0, metrics={_KEY: False}),
                _metric_event(rollout_id=6, attempt=0, metrics={_KEY: "4.0"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(4, 3.0)]


class TestAssertMetricsClassified:
    def test_an_unclassified_namespace_fails_closed(self, dump_dir: str) -> None:
        """A final-attempt metric outside every declared namespace makes classification fail closed."""
        _write_metric_events(
            dump_dir,
            [_metric_event(rollout_id=0, attempt=0, metrics={"unexpected/value": 1.0})],
        )

        with pytest.raises(AssertionError, match="unexpected/value"):
            assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))


class TestAssertEveryMetricIsClassified:
    def test_unknown_final_attempt_namespaces_are_reported(self, dump_dir: str) -> None:
        """Sorted unknown final-attempt keys are reported while superseded unknown keys are ignored."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0, "zeta/value": 3.0, "alpha/value": 4.0},
                ),
            ],
        )

        with pytest.raises(AssertionError, match=r"metrics \['alpha/value', 'zeta/value'\]"):
            assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))

    def test_compared_and_ignored_final_attempt_namespaces_are_accepted(self, dump_dir: str) -> None:
        """Final-attempt keys covered by either compared or ignored prefixes are accepted."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0},
                ),
            ],
        )

        assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))


class TestReadRolloutCompletionTimes:
    def test_final_identified_rollouts_are_sorted_by_completion_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Completion times contain identified final attempts ordered by their timestamps."""
        timestamps = iter(
            [
                datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            ]
        )

        class _SequencedDatetime(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                return next(timestamps)

        monkeypatch.setattr("miles.utils.audit_utils.event_logger.logger.datetime", _SequencedDatetime)
        dump_dir = tmp_path / "run"
        event_logger = EventLogger(log_dir=dump_dir / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
        for rollout_id, attempt in [(20, 0), (None, 0), (10, 0), (20, 1)]:
            event_logger.log(
                MetricEvent,
                {"rollout_id": rollout_id, "attempt": attempt, "metrics": {}},
                print_log=False,
            )
        event_logger.close()

        assert read_rollout_completion_times(str(dump_dir)) == [
            (20, datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
            (10, datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)),
        ]


class TestReadMetricSeries:
    def test_only_numeric_values_from_the_final_attempt_are_returned(self, dump_dir: str) -> None:
        """Only numeric final-attempt values survive, with integers normalized to floats."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={_KEY: 99.0}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 1}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 2.5}),
                _metric_event(rollout_id=1, attempt=0, metrics={_KEY: True}),
                _metric_event(rollout_id=2, attempt=0, metrics={"train/loss": 3.0}),
                _metric_event(rollout_id=3, attempt=0, metrics={_KEY: "4.0"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(0, 1.0), (0, 2.5)]

    def test_only_numeric_values_from_final_attempts_form_the_series(self, dump_dir: str) -> None:
        """Retries and nonnumeric payloads must not contaminate a numeric comparison series."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={_KEY: 1.0}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 2}),
                _metric_event(rollout_id=1, attempt=0, metrics={_KEY: True}),
                _metric_event(rollout_id=2, attempt=0, metrics={_KEY: "unknown"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(0, 2.0)]


class TestAssertEveryMetricIsClassified:
    def test_unknown_final_attempt_namespaces_are_reported(self, dump_dir: str) -> None:
        """Sorted unknown final-attempt keys are reported while superseded unknown keys are ignored."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0, "zeta/value": 3.0, "alpha/value": 4.0},
                ),
            ],
        )

        with pytest.raises(AssertionError, match=r"metrics \['alpha/value', 'zeta/value'\]"):
            assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))

    def test_compared_and_ignored_final_attempt_namespaces_are_accepted(self, dump_dir: str) -> None:
        """Final-attempt keys covered by either compared or ignored prefixes are accepted."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0},
                ),
            ],
        )

        assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))


class TestAssertMetricsClassified:
    def test_an_unclassified_namespace_fails_closed(self, dump_dir: str) -> None:
        """A newly logged namespace cannot silently escape a comparison that claims complete coverage."""
        _write_metric_events(dump_dir, [_metric_event(rollout_id=0, attempt=0, metrics={_KEY: 0.5})])

        with pytest.raises(AssertionError, match="belong to no namespace"):
            assert_metrics_classified(dump_dir, compared=("rollout/",), ignored=("perf/",))

    @pytest.mark.parametrize("unusable_value", [float("nan"), float("inf"), float("-inf"), 0.0])
    def test_nonfinite_and_zero_values_do_not_count_as_trained_rollouts(
        self, dump_dir: str, unusable_value: float
    ) -> None:
        """The gradient wrapper rejects rollouts whose train/grad_norm is non-finite or zero."""
        _write_metrics(dump_dir, [(0, 0.5), (1, unusable_value)])

        with pytest.raises(AssertionError, match="target: train/grad_norm is finite and non-zero in only 1"):
            assert_gradients_nonzero(side="target", dump_dir=dump_dir, min_trained_rollouts=2)


@pytest.fixture
def dump_dir(tmp_path) -> str:
    return str(tmp_path / "run")


def _write_metric_events(dump_dir: str, events: list[MetricEvent]) -> None:
    event_logger = EventLogger(log_dir=Path(dump_dir) / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
    for event in events:
        partial = event.model_dump(exclude={"timestamp", "source", "type"})
        event_logger.log(MetricEvent, partial, print_log=False)
    event_logger.close()


def _write_metric_events(dump_dir: str, events: list[MetricEvent]) -> None:
    event_logger = EventLogger(log_dir=Path(dump_dir) / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
    for event in events:
        partial = event.model_dump(exclude={"timestamp", "source", "type"})
        event_logger.log(MetricEvent, partial, print_log=False)
    event_logger.close()
