from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _injected, _request, _requested, _write_sut_lines
from tests.utils.soak.core.events import SoakEvent, SoakEvidenceArchivedEvent
from tests.utils.soak.ft.checkers.progress_windows import (
    _compute_fault_progress_windows,
    assert_faults_span_progress_windows,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_BASE = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


class TestComputeCrashedRollouts:
    def test_two_crashes_before_any_rollout_finished_share_one_window(self) -> None:
        """Both landing before the first generation is the vacuous run this scenario has to reject."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE, _BASE + timedelta(seconds=10)], rollout_completions=[(0, _BASE + timedelta(hours=1))]
        )

        assert crashed == {0}

    def test_crashes_on_either_side_of_a_finished_rollout_are_two_windows(self) -> None:
        """This is the run the scenario is meant to produce: crashes spread across the loss curve."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE, _BASE + timedelta(seconds=20)],
            rollout_completions=[(0, _BASE + timedelta(seconds=10))],
        )

        assert crashed == {0, 1}

    def test_a_run_with_no_crashes_has_no_windows(self) -> None:
        """An empty result must not read as coverage; the caller's floor is what rejects it."""
        assert _compute_fault_progress_windows(injected_at=[], rollout_completions=[(0, _BASE)]) == set()

    def test_repeated_metrics_from_one_rollout_count_as_one_completed_rollout(self) -> None:
        """Repeated metric events must not advance an injection by multiple rollout windows."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE + timedelta(seconds=20)],
            rollout_completions=[
                (0, _BASE + timedelta(seconds=10)),
                (0, _BASE + timedelta(seconds=15)),
            ],
        )

        assert crashed == {1}


def _write_completions(events_dir: Path, *, rollout_at: dict[int, float], component: str = "rollout_executor") -> None:
    events_dir.mkdir(parents=True, exist_ok=True)
    _write_sut_lines(
        events_dir / f"{component}.jsonl",
        [
            MetricEvent(
                timestamp=_at(at), source=SimpleProcessIdentity(component=component), rollout_id=rollout_id, metrics={}
            )
            for rollout_id, at in rollout_at.items()
        ],
    )


def _faults(*, kind: str = "rollout", applied_at: list[float]) -> list[SoakEvent]:
    return [
        event
        for index, at in enumerate(applied_at)
        for event in _injected(
            _request(_cell_target(kind=kind), form_name="kill", request_id=f"{kind}-{index}"), start=at - 1
        )
    ]


class TestAssertFaultsSpanProgressWindows:
    def test_rollout_faults_separated_by_a_completed_rollout_pass(self, tmp_path: Path) -> None:
        """Two applied rollout faults on either side of a completion span two windows."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})

        assert_faults_span_progress_windows(_faults(applied_at=[5, 15]), dump_dir=str(tmp_path))

    def test_rollout_faults_inside_one_window_are_rejected(self, tmp_path: Path) -> None:
        """Faults that all land before the first completion exercise only one window."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})

        with pytest.raises(AssertionError, match="progress windows"):
            assert_faults_span_progress_windows(_faults(applied_at=[5, 6]), dump_dir=str(tmp_path))

    def test_a_fault_at_the_completion_instant_belongs_to_the_next_window(self, tmp_path: Path) -> None:
        """A rollout finished at the fault time already counts as completed."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})

        assert_faults_span_progress_windows(_faults(applied_at=[5, 10]), dump_dir=str(tmp_path))

    def test_trainer_faults_do_not_count(self, tmp_path: Path) -> None:
        """Only rollout faults are spread over rollout progress windows."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})
        events = [*_faults(applied_at=[5]), *_faults(kind="actor", applied_at=[15])]

        with pytest.raises(AssertionError, match="progress windows"):
            assert_faults_span_progress_windows(events, dump_dir=str(tmp_path))

    def test_requests_that_never_applied_do_not_count(self, tmp_path: Path) -> None:
        """A draw that never landed occupies no window."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})
        unapplied = _request(_cell_target(kind="rollout"), form_name="kill", request_id="unapplied")
        events = [*_faults(applied_at=[5]), _requested(unapplied, at=_at(15))]

        with pytest.raises(AssertionError, match="progress windows"):
            assert_faults_span_progress_windows(events, dump_dir=str(tmp_path))

    def test_completions_of_other_processes_are_not_rollout_completions(self, tmp_path: Path) -> None:
        """Only the rollout executor's metrics mark finished rollouts."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10}, component="main")

        with pytest.raises(AssertionError, match="progress windows"):
            assert_faults_span_progress_windows(_faults(applied_at=[5, 15]), dump_dir=str(tmp_path))

    def test_the_archived_training_events_are_read_instead_of_the_live_dump(self, tmp_path: Path) -> None:
        """After archiving, the checker judges the archived copy and ignores later live writes."""
        archived = tmp_path / "evidence" / "sources" / "training_events" / EVENTS_DIRNAME
        _write_completions(archived, rollout_at={0: 10})
        (tmp_path / "dump" / EVENTS_DIRNAME).mkdir(parents=True)
        events = [
            *_faults(applied_at=[5, 15]),
            SoakEvidenceArchivedEvent(
                timestamp=_at(20), sources={"training_events": archived}, missing_sources=[], sha256_of_file={}
            ),
        ]

        assert_faults_span_progress_windows(events, dump_dir=str(tmp_path / "dump"))

    def test_training_events_archived_as_missing_are_refused(self, tmp_path: Path) -> None:
        """A missing archive cannot be replaced by whatever the live dump holds now."""
        _write_completions(tmp_path / EVENTS_DIRNAME, rollout_at={0: 10})
        events = [
            *_faults(applied_at=[5, 15]),
            SoakEvidenceArchivedEvent(
                timestamp=_at(20), sources={}, missing_sources=["training_events"], sha256_of_file={}
            ),
        ]

        with pytest.raises(AssertionError, match="Missing archived soak evidence"):
            assert_faults_span_progress_windows(events, dump_dir=str(tmp_path))
