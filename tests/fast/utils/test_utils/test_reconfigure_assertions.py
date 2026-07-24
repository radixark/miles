from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.test_utils.reconfigure_assertions import (
    ReconfigureInfo,
    assert_reconfigure_events,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)

_SHRINK_PARTIAL: dict[str, Any] = dict(
    rollout_id=2,
    quorum_id=1,
    src_cell_index=None,
    healed_cell_indices=[],
    alive_cell_indices_after=[0],
)
_HEALING_PARTIAL: dict[str, Any] = dict(
    rollout_id=3,
    quorum_id=2,
    src_cell_index=0,
    healed_cell_indices=[1],
    alive_cell_indices_after=[0, 1],
)

_SHRINK_EXPECTED = ReconfigureInfo(
    rollout_id=2, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]
)
_HEALING_EXPECTED = ReconfigureInfo(
    rollout_id=3, src_cell_index=0, healed_cell_indices=[1], alive_cell_indices_after=[0, 1]
)


def _write_events(log_dir: Path, partials: list[dict[str, Any]]) -> None:
    event_logger = EventLogger(log_dir=log_dir, source=MainProcessIdentity())
    for partial in partials:
        event_logger.log(CellReconfigureEvent, partial, print_log=False)
    event_logger.close()


class TestLoadReconfigureEvents:
    def test_filters_other_event_types_and_preserves_order(self, tmp_path: Path) -> None:
        """Only CellReconfigureEvents are returned, in file (emission) order."""
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(CellReconfigureEvent, _SHRINK_PARTIAL, print_log=False)
        event_logger.log(TrainGroupStepEndEvent, dict(rollout_id=2, cell_outcomes={}), print_log=False)
        event_logger.log(CellReconfigureEvent, _HEALING_PARTIAL, print_log=False)
        event_logger.close()

        events = load_reconfigure_events(tmp_path)

        assert [e.rollout_id for e in events] == [2, 3]
        assert all(isinstance(e, CellReconfigureEvent) for e in events)

    def test_empty_dir_returns_no_events(self, tmp_path: Path) -> None:
        """A directory without any JSONL files yields an empty event list."""
        assert load_reconfigure_events(tmp_path) == []


class TestAssertReconfigureEvents:
    def test_passes_on_exact_sequence(self, tmp_path: Path) -> None:
        """An exactly matching shrink+healing sequence with contiguous quorum ids passes."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL])

        assert_reconfigure_events(tmp_path, expected=[_SHRINK_EXPECTED, _HEALING_EXPECTED])

    def test_passes_on_empty_expectation(self, tmp_path: Path) -> None:
        """Expecting zero reconfigures passes when no events were emitted."""
        assert_reconfigure_events(tmp_path, expected=[])

    def test_missing_healing_fails_sequence_check(self, tmp_path: Path) -> None:
        """A run that never healed fails the exact-sequence comparison (expected healing, got nothing)."""
        _write_events(tmp_path, [])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])

    def test_unexpected_extra_healing_fails(self, tmp_path: Path) -> None:
        """A healing event in a run expected to have none fails the exact-sequence comparison."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, quorum_id=1)])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[])

    def test_wrong_rollout_id_fails_sequence_check(self, tmp_path: Path) -> None:
        """A healing at the wrong rollout fails the exact-sequence comparison."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, rollout_id=9, quorum_id=1)])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])


class TestAssertSoakReconfigureEvents:
    def test_passes_when_enough_injections_and_healings(self, tmp_path: Path) -> None:
        """>=2 successful injections with >=2 healing events pass the soak witness."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL, dict(_HEALING_PARTIAL, rollout_id=5, quorum_id=3)])

        assert_soak_reconfigure_events(tmp_path, num_successful_injections=2)

    def test_fails_when_no_injections(self, tmp_path: Path) -> None:
        """Zero successful injections means no fault tolerance was exercised, so the witness fails."""
        with pytest.raises(AssertionError, match="proved too little"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=0)

    def test_fails_when_too_few_injections(self, tmp_path: Path) -> None:
        """A single injection is below the soak minimum even when healing events are present."""
        _write_events(tmp_path, [_HEALING_PARTIAL, dict(_HEALING_PARTIAL, rollout_id=5, quorum_id=3)])

        with pytest.raises(AssertionError, match="proved too little"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=1)

    def test_fails_when_too_few_healings(self, tmp_path: Path) -> None:
        """Enough injections but fewer than the required healing events fail the witness."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL])

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_soak_reconfigure_events(tmp_path, num_successful_injections=3)
