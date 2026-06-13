from pathlib import Path
from typing import Any

import pytest

from miles.utils.event_logger.logger import EventLogger
from miles.utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
from miles.utils.process_identity import MainProcessIdentity
from miles.utils.test_utils.reconfigure_assertions import (
    ExpectedReconfigure,
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

_SHRINK_EXPECTED = ExpectedReconfigure(
    rollout_id=2, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]
)
_HEALING_EXPECTED = ExpectedReconfigure(
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

    def test_missing_healing_fails_with_explicit_count_message(self, tmp_path: Path) -> None:
        """A run that never healed fails with an 'expected N healing event(s), got M' message."""
        _write_events(tmp_path, [])

        with pytest.raises(AssertionError, match=r"expected 1 healing event\(s\), got 0"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])

    def test_unexpected_extra_healing_fails(self, tmp_path: Path) -> None:
        """A healing event in a run expected to have none fails the healing count check."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, quorum_id=1)])

        with pytest.raises(AssertionError, match=r"expected 0 healing event\(s\), got 1"):
            assert_reconfigure_events(tmp_path, expected=[])

    def test_wrong_rollout_id_fails_sequence_check(self, tmp_path: Path) -> None:
        """A healing at the wrong rollout fails the exact-sequence comparison."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, rollout_id=9, quorum_id=1)])

        with pytest.raises(AssertionError, match="sequence mismatch"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])

    def test_quorum_gap_fails(self, tmp_path: Path) -> None:
        """A quorum id gap (a hidden failed reconfigure attempt) fails even if shapes match."""
        _write_events(tmp_path, [dict(_HEALING_PARTIAL, quorum_id=2)])

        with pytest.raises(AssertionError, match="quorum ids"):
            assert_reconfigure_events(tmp_path, expected=[_HEALING_EXPECTED])


class TestAssertSoakReconfigureEvents:
    _FINAL_ROLLOUT_ID = 29

    def test_passes_when_injections_led_to_healing_and_full_membership(self, tmp_path: Path) -> None:
        """Injections with a healing that restores full membership pass."""
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL])

        assert_soak_reconfigure_events(
            tmp_path, num_successful_injections=1, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
        )

    def test_passes_with_no_injections_and_no_events(self, tmp_path: Path) -> None:
        """Zero injections and zero events is a legitimate (quiet) soak outcome."""
        assert_soak_reconfigure_events(
            tmp_path, num_successful_injections=0, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
        )

    def test_passes_with_single_trailing_shrink_at_final_rollout(self, tmp_path: Path) -> None:
        """A fault inside the final rollout's train() leaves one trailing shrink, which is tolerated."""
        trailing_shrink = dict(_SHRINK_PARTIAL, quorum_id=3, rollout_id=self._FINAL_ROLLOUT_ID)
        _write_events(tmp_path, [_SHRINK_PARTIAL, _HEALING_PARTIAL, trailing_shrink])

        assert_soak_reconfigure_events(
            tmp_path, num_successful_injections=2, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
        )

    def test_fails_when_injections_but_no_healing(self, tmp_path: Path) -> None:
        """Successful injections without any healing event fail the witness."""
        _write_events(tmp_path, [_SHRINK_PARTIAL])

        with pytest.raises(AssertionError, match="no healing event"):
            assert_soak_reconfigure_events(
                tmp_path, num_successful_injections=2, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
            )

    def test_fails_when_injections_but_no_healing_even_with_trailing_shrink_at_final_rollout(
        self, tmp_path: Path
    ) -> None:
        """The trailing-shrink tolerance never excuses a run with injections but zero healings."""
        _write_events(tmp_path, [dict(_SHRINK_PARTIAL, rollout_id=self._FINAL_ROLLOUT_ID)])

        with pytest.raises(AssertionError, match="no healing event"):
            assert_soak_reconfigure_events(
                tmp_path, num_successful_injections=1, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
            )

    def test_fails_when_trailing_shrink_is_not_at_final_rollout(self, tmp_path: Path) -> None:
        """A trailing shrink at any rollout other than the final one fails the fully-healed check."""
        _write_events(tmp_path, [_HEALING_PARTIAL, dict(_SHRINK_PARTIAL, quorum_id=3, rollout_id=9)])

        with pytest.raises(AssertionError, match="end fully healed"):
            assert_soak_reconfigure_events(
                tmp_path, num_successful_injections=1, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
            )

    def test_fails_with_two_trailing_shrinks_even_at_final_rollout(self, tmp_path: Path) -> None:
        """Only one trailing shrink is tolerated; two in a row fail even when both carry the final rollout id."""
        first_shrink = dict(_SHRINK_PARTIAL, quorum_id=3, rollout_id=self._FINAL_ROLLOUT_ID)
        second_shrink = dict(
            _SHRINK_PARTIAL, quorum_id=4, rollout_id=self._FINAL_ROLLOUT_ID, alive_cell_indices_after=[]
        )
        _write_events(tmp_path, [_HEALING_PARTIAL, first_shrink, second_shrink])

        with pytest.raises(AssertionError, match="end fully healed"):
            assert_soak_reconfigure_events(
                tmp_path, num_successful_injections=2, num_cells=2, final_rollout_id=self._FINAL_ROLLOUT_ID
            )
