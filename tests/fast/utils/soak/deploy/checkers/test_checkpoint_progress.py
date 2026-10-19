import shutil
from pathlib import Path

import pytest
from tests.fast.utils.soak.deploy.deploy_fakes import (
    _deployment_target,
    _hot_restart_request,
    _landed_take_over,
    _requested_take_over,
    _restamped,
    _roll_log_aside,
    _write_finished_steps,
)
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _request, _requested
from tests.utils.deploy.hot_restart.evidence import HotRestartRecord
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.deploy.checkers.checkpoint_progress import (
    MAX_REDONE_STEPS_PER_TAKE_OVER,
    MIN_HOT_RESTARTS,
    SAVE_INTERVAL,
    assert_checkpoints_advanced_between_takeovers,
    assert_take_over_loss_within_save_interval,
    assert_take_overs_resumed_within_save_interval,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME


def _take_over(
    request_id: str, *, saved: int | None, saved_after: int | None, start: float, applied: bool = True
) -> list[SoakEvent]:
    before = _deployment_target(saved_iteration=saved)
    request = _hot_restart_request(before, request_id=request_id)
    events: list[SoakEvent] = [_requested_take_over(request, at=_at(start))]
    if applied:
        after = _restamped(before, request_id, saved_iteration=saved_after)
        events.append(_landed_take_over(request, after=after, at=_at(start + 1)))
    return events


def _record(*, index: int = 0, saved: int | None, frozen: int) -> HotRestartRecord:
    return HotRestartRecord(index=index, saved_iteration_at_trigger=saved, frozen_rollout_id=frozen)


def _records(count: int) -> list[HotRestartRecord]:
    return [_record(index=index, saved=0, frozen=0) for index in range(count)]


class TestAssertCheckpointsAdvancedBetweenTakeovers:
    def test_take_overs_each_drawn_after_a_newer_checkpoint_pass(self) -> None:
        """Each take-over must find a checkpoint newer than the one left when the previous one landed."""
        events = [
            *_take_over("a", saved=0, saved_after=3, start=0),
            *_take_over("b", saved=4, saved_after=4, start=10),
        ]

        assert_checkpoints_advanced_between_takeovers(events)

    @pytest.mark.parametrize("second_saved", [3, None], ids=["same_checkpoint", "no_checkpoint"])
    def test_a_take_over_without_a_newer_checkpoint_fails(self, second_saved: int | None) -> None:
        """Resuming from the checkpoint the previous take-over already saw would redo its whole window."""
        events = [
            *_take_over("a", saved=0, saved_after=3, start=0),
            *_take_over("b", saved=second_saved, saved_after=6, start=10),
        ]

        with pytest.raises(AssertionError, match="Takeover b lacks a new checkpoint"):
            assert_checkpoints_advanced_between_takeovers(events)

    def test_the_newer_side_of_a_landing_sets_the_bar(self) -> None:
        """A checkpoint saved while the take-over was landing already counts as seen by the next one."""
        events = [
            *_take_over("a", saved=5, saved_after=2, start=0),
            *_take_over("b", saved=5, saved_after=8, start=10),
        ]

        with pytest.raises(AssertionError, match="saved=5, previous=5"):
            assert_checkpoints_advanced_between_takeovers(events)

    def test_a_first_take_over_before_any_save_fails(self) -> None:
        """The first take-over also needs some checkpoint to resume from."""
        events = [
            *_take_over("a", saved=None, saved_after=3, start=0),
            *_take_over("b", saved=4, saved_after=5, start=10),
        ]

        with pytest.raises(AssertionError, match="Takeover a"):
            assert_checkpoints_advanced_between_takeovers(events)

    def test_fewer_applied_take_overs_than_the_minimum_fail(self) -> None:
        """A run that was taken over once, or requested more but never landed, proves too little."""
        events = [
            *_take_over("a", saved=0, saved_after=3, start=0),
            *_take_over("b", saved=4, saved_after=None, start=10, applied=False),
        ]

        with pytest.raises(AssertionError, match=f"at least {MIN_HOT_RESTARTS} applied takeovers, got 1"):
            assert_checkpoints_advanced_between_takeovers(events)

    def test_unapplied_and_other_form_actions_are_ignored(self) -> None:
        """Requests that never landed and cell faults neither count nor set the checkpoint bar."""
        cell_request = _request(_cell_target(), form_name="inject_fault:kill_process", request_id="cell")
        events = [
            *_take_over("a", saved=0, saved_after=3, start=0),
            *_take_over("x", saved=None, saved_after=None, start=5, applied=False),
            _requested(cell_request, at=_at(6)),
            _applied(cell_request, at=_at(7)),
            *_take_over("b", saved=4, saved_after=4, start=10),
        ]

        assert_checkpoints_advanced_between_takeovers(events)


class TestAssertTakeOverLossWithinSaveInterval:
    def test_the_bound_is_one_save_interval_plus_the_step_in_flight(self) -> None:
        """The loss bound must follow the save interval the run is installed with."""
        assert MAX_REDONE_STEPS_PER_TAKE_OVER == SAVE_INTERVAL + 1

    @pytest.mark.parametrize(
        ("saved", "frozen"),
        [
            pytest.param(5, 5, id="nothing_redone"),
            pytest.param(2, 2 + MAX_REDONE_STEPS_PER_TAKE_OVER, id="at_the_bound"),
            pytest.param(None, MAX_REDONE_STEPS_PER_TAKE_OVER - 1, id="before_first_save"),
        ],
    )
    def test_losses_within_the_bound_pass(self, saved: int | None, frozen: int) -> None:
        """Between zero and the bound, whatever the timing of the draw, a take-over costs an acceptable redo."""
        assert_take_over_loss_within_save_interval([_record(saved=saved, frozen=frozen)])

    @pytest.mark.parametrize(
        ("saved", "frozen"),
        [
            pytest.param(2, 3 + MAX_REDONE_STEPS_PER_TAKE_OVER, id="past_the_bound"),
            pytest.param(None, MAX_REDONE_STEPS_PER_TAKE_OVER, id="before_first_save_past_the_bound"),
            pytest.param(6, 5, id="checkpoint_ahead_of_progress"),
        ],
    )
    def test_losses_outside_the_bound_fail(self, saved: int | None, frozen: int) -> None:
        """Redoing more than the bound, or a checkpoint ahead of progress, means the resume point was wrong."""
        with pytest.raises(AssertionError, match="threw away"):
            assert_take_over_loss_within_save_interval([_record(index=1, saved=saved, frozen=frozen)])

    def test_every_record_is_checked(self) -> None:
        """A single bad take-over among good ones must still fail."""
        records = [_record(index=0, saved=5, frozen=5), _record(index=1, saved=0, frozen=9)]

        with pytest.raises(AssertionError, match="take-over 1"):
            assert_take_over_loss_within_save_interval(records)


class TestAssertTakeOversResumedWithinSaveInterval:
    def test_a_take_over_that_redid_only_what_its_checkpoint_missed_passes(self, tmp_path: Path) -> None:
        """What a take-over cost is the log it replaced minus the prefix the run that followed kept."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 6))

        assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_resumed_further_back_than_a_save_interval_fails(self, tmp_path: Path) -> None:
        """Measuring the resume point off the trigger's tracker would pass a run that reloaded an older save."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=[])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))

        with pytest.raises(AssertionError, match="redid 6 step"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_left_no_rolled_back_log_fails(self, tmp_path: Path) -> None:
        """A take-over whose log is missing hides the very steps this assertion counts."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))

        with pytest.raises(AssertionError, match="rolls the log it replaced aside"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_more_rolled_aside_logs_than_take_overs_fail(self, tmp_path: Path) -> None:
        """An extra rolled-aside log means a restart nobody recorded replaced the run."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120100", kept=range(2))

        with pytest.raises(AssertionError, match="rolls the log it replaced aside"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_carried_a_hole_over_fails(self, tmp_path: Path) -> None:
        """A run resumes from one checkpoint, so what survives a take-over is a prefix and never a hole."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=[0, 2])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="carried the steps"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_fired_during_the_catch_up_is_read_off_the_log_it_rolled_aside(self, tmp_path: Path) -> None:
        """A take-over during the catch-up leaves a log reaching a lower step than the log before it."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120100", kept=range(7))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(7, 12))

        assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_a_take_over_fired_during_the_catch_up_is_measured_against_its_own_log(self, tmp_path: Path) -> None:
        """Ordering the rolled-aside logs by how far they trained blames the wrong take-over for the redone steps."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120100", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 12))

        with pytest.raises(AssertionError, match="take-over 1 replaced a log that had reached step 8"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_two_take_overs_that_rolled_their_logs_aside_in_the_same_second_fail(self, tmp_path: Path) -> None:
        """Which log a take-over replaced is read off when it was rolled aside, so a tie leaves it unpinned."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 5))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(3))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="same second"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_a_rolled_aside_log_without_a_timestamp_name_fails(self, tmp_path: Path) -> None:
        """A trash directory not named by its roll-aside time cannot be ordered against the others."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        (tmp_path / EVENTS_DIRNAME).rename(tmp_path / ".trash_manual")
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 4))

        with pytest.raises(AssertionError, match="does not name the moment"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_log_describing_a_step_twice_fails(self, tmp_path: Path) -> None:
        """A take-over rolls the log back before redoing anything, so a duplicate step means it did not."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _roll_log_aside(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [2, 3])
        shutil.copy(tmp_path / EVENTS_DIRNAME / "step-3.jsonl", tmp_path / EVENTS_DIRNAME / "step-3-again.jsonl")

        with pytest.raises(AssertionError, match="more than once"):
            assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))
