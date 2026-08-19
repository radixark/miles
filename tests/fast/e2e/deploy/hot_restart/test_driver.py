import threading
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import driver as driver_module
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import HotRestartDriver, ScheduledFreeze, compute_freeze_plan
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, RunProgress

from miles.utils.test_utils.ft_test_actions import SLEEP_FOREVER_AT_END_ACTION, FTTestAction


_CHECKPOINTED: ScheduledFreeze = ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1)
_FROM_SCRATCH: ScheduledFreeze = ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None)
_SCHEDULE: tuple[ScheduledFreeze, ...] = (
    ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
    ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
)


class TestScheduledFreeze:
    def test_a_freeze_past_the_checkpoint_it_resumes_from_is_accepted(self):
        """This is the whole window: a checkpoint to resume from, and a step past it to redo."""
        assert ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1).saved_iteration == 1

    def test_a_freeze_before_the_first_save_is_accepted(self):
        """A run holding no checkpoint at all resumes from the reference weights, which is a scenario of its own."""
        assert ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None).saved_iteration is None

    def test_a_freeze_at_the_step_its_own_checkpoint_covers_is_refused(self):
        """Such a take-over resumes exactly where the script it replaced stood, so it redoes nothing."""
        with pytest.raises(AssertionError, match="nothing would be redone"):
            ScheduledFreeze(frozen_rollout_id=2, saved_iteration=2)

    def test_a_freeze_before_the_run_starts_is_refused(self):
        """A run frozen before step 0 has trained nothing, so a take-over of it costs nothing."""
        with pytest.raises(AssertionError, match="a run starts at step 0"):
            ScheduledFreeze(frozen_rollout_id=-1, saved_iteration=None)


class TestTheFreezePlan:
    def test_the_plan_arms_the_sleep_forever_action_at_the_pinned_step(self):
        """The run reads this plan, and it is the only thing that decides where the run stands still."""
        [action] = [FTTestAction(**one) for one in compute_freeze_plan(3)]

        assert action == FTTestAction(at_rollout=3, action=SLEEP_FOREVER_AT_END_ACTION)

    def test_a_run_with_no_freeze_left_carries_an_empty_plan(self):
        """The last relaunch has to train to the end, and a plan repeating the old step would freeze it again."""
        assert compute_freeze_plan(None) == []


def _driver(tmp_path: Path, **overrides: Any) -> HotRestartDriver:
    kwargs: dict[str, Any] = dict(
        relaunch=lambda _frozen_rollout_id: None,
        checkpoint_dir=tmp_path / "checkpoints",
        events_dir=tmp_path / "events",
        schedule=_SCHEDULE,
        poll_interval_seconds=0.0,
    )
    kwargs.update(overrides)
    return HotRestartDriver(**kwargs)


def _join_relaunches(driver: HotRestartDriver) -> None:
    for thread in driver._relaunch_threads:
        thread.join(timeout=30.0)


def _install_progress(monkeypatch, reported: list[RunProgress]) -> None:
    remaining = list(reported)
    monkeypatch.setattr(
        driver_module, "read_run_progress", lambda **_kwargs: remaining.pop(0) if remaining else reported[-1]
    )


class TestHotRestartDriverStart:
    def test_a_dump_directory_holding_a_previous_run_is_refused(self, tmp_path):
        """Resuming from what a previous run left would compare a run that never started from nothing."""
        driver = _driver(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        with pytest.raises(AssertionError, match="before this run was even installed"):
            driver.start()

    def test_a_run_nothing_would_freeze_is_refused(self, tmp_path):
        """The driver waits for a frozen run, and a run with no plan never stands still for it."""
        with pytest.raises(AssertionError, match="nothing freezes"):
            _driver(tmp_path, schedule=())


class TestHotRestartDriverProgressGuard:
    def test_a_run_whose_event_log_shrank_before_any_restart_fails(self, tmp_path):
        """Nothing but a take-over rolls the log back, so this is a run reading someone else's dumps."""
        driver = _driver(tmp_path)
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))

        with pytest.raises(AssertionError, match="lost work"):
            driver._assert_no_step_lost_outside_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
            )

    def test_the_rollback_a_take_over_performs_is_not_read_as_lost_work(self, tmp_path):
        """The log going back to the checkpoint is the very thing a hot restart is supposed to do."""
        driver = _driver(tmp_path)
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))
        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))

        driver._assert_no_step_lost_outside_take_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))


class TestTheFreezeATakeOverWaitsFor:
    def test_a_run_that_has_finished_nothing_is_not_frozen_yet(self, tmp_path):
        """Polling starts before the pods are even up."""
        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=None)
        )

    def test_a_run_short_of_the_pinned_step_is_not_frozen_yet(self, tmp_path):
        """Taking a run over one step early would resume it from another checkpoint than the pinned one."""
        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
        )

    def test_the_run_standing_at_the_pinned_step_is_the_frozen_one(self, tmp_path):
        """The run sleeps from here on, so this is where it stands until something replaces it."""
        assert _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )

    def test_a_run_that_trained_past_the_pinned_step_fails_instead_of_firing(self, tmp_path):
        """The freeze is what makes the take-over exact, and a run that moved on was never frozen."""
        with pytest.raises(AssertionError, match="never fired"):
            _driver(tmp_path)._stands_frozen_at(
                _CHECKPOINTED, progress=RunProgress(last_saved_iteration=3, last_finished_rollout_id=3)
            )


class TestTheRecordATakeOverLeaves:
    def test_the_record_is_the_pinned_take_over_the_run_was_frozen_for(self, tmp_path):
        """The comparison reads the window off this record, and the freeze makes it the scheduled one."""
        record = _driver(tmp_path)._compute_record(
            index=0,
            scheduled=_CHECKPOINTED,
            progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
        )

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2)

    def test_a_run_holding_no_checkpoint_is_recorded_as_such(self, tmp_path):
        """Only the record tells the comparison which of the two take-over paths this dump describes."""
        record = _driver(tmp_path)._compute_record(
            index=0,
            scheduled=_FROM_SCRATCH,
            progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=1),
        )

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=1)

    def test_a_run_saving_at_another_pace_than_the_scenario_pinned_records_nothing(self, tmp_path):
        """The pinned save is what the take-over rolls back to, so a run that wrote another one lands elsewhere."""
        with pytest.raises(AssertionError, match="save cadence"):
            _driver(tmp_path)._compute_record(
                index=0,
                scheduled=_CHECKPOINTED,
                progress=RunProgress(last_saved_iteration=2, last_finished_rollout_id=2),
            )

    def test_a_run_that_saved_before_the_take_over_of_a_run_holding_nothing_records_nothing(self, tmp_path):
        """A save here means the take-over walks the path the checkpointed mode already covers."""
        with pytest.raises(AssertionError, match="save cadence"):
            _driver(tmp_path)._compute_record(
                index=0,
                scheduled=_FROM_SCRATCH,
                progress=RunProgress(last_saved_iteration=0, last_finished_rollout_id=1),
            )


class TestTheTakeOverLoop:
    def test_nothing_is_relaunched_until_the_run_stands_at_the_pinned_step(self, tmp_path, monkeypatch):
        """A take-over of a run still training would land wherever the poll happened to fall."""
        armed: list[int | None] = []
        driver = _driver(tmp_path, relaunch=armed.append)
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)])

        stop_event = threading.Event()
        stop_event.set()
        driver._drive(stop_event)

        assert driver.records == []
        assert armed == []

    def test_each_relaunch_arms_the_freeze_the_next_take_over_waits_for(self, tmp_path, monkeypatch):
        """A relaunch repeating the old step would freeze the run again where it was already taken over."""
        armed: list[int | None] = []
        driver = _driver(tmp_path, relaunch=armed.append)
        _install_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=3, last_finished_rollout_id=4),
            ],
        )

        driver._drive(threading.Event())
        _join_relaunches(driver)

        assert driver.records == [
            HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),
            HotRestartRecord(index=1, saved_iteration_at_trigger=3, frozen_rollout_id=4),
        ]
        assert armed == [4, None]
        driver.assert_all_restarts_happened()

    def test_the_loop_ends_once_the_last_take_over_was_triggered(self, tmp_path, monkeypatch):
        """A run read again after its final take-over would fail for training past its last freeze."""
        armed: list[int | None] = []
        driver = _driver(
            tmp_path, relaunch=armed.append, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),)
        )
        _install_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=5, last_finished_rollout_id=5),
            ],
        )

        driver._drive(threading.Event())
        _join_relaunches(driver)

        assert armed == [None]
        assert len(driver.records) == 1

    def test_a_run_that_never_reaches_the_pinned_step_runs_out_of_time(self, tmp_path, monkeypatch):
        """Waiting forever would leave the suite hanging on a run whose freeze never fired."""
        driver = _driver(tmp_path, freeze_timeout_seconds=0.0)
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)])

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="waited 0.0s"):
            driver.assert_all_restarts_happened()

    def test_a_read_that_failed_is_retried_rather_than_ending_the_run(self, tmp_path, monkeypatch):
        """A dump directory on shared storage answers late now and then, and that is not a verdict."""
        reads: list[int] = []

        def read(**_kwargs):
            reads.append(1)
            if len(reads) == 1:
                raise OSError("shared storage said no")
            return RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)

        driver = _driver(tmp_path, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),))
        monkeypatch.setattr(driver_module, "read_run_progress", read)

        driver._drive(threading.Event())
        _join_relaunches(driver)

        assert len(driver.records) == 1


class TestHotRestartDriverVerdict:
    def test_a_run_that_ended_before_every_take_over_landed_fails(self, tmp_path):
        """A run replaced once when the scenario pinned two proves half of what it claims."""
        driver = _driver(tmp_path)
        driver.records.append(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2))

        with pytest.raises(AssertionError, match="1 of 2 hot restart"):
            driver.assert_all_restarts_happened()

    def test_a_relaunch_that_raised_is_reported_rather_than_swallowed(self, tmp_path):
        """A take-over nothing installed leaves a run the comparison would read as untouched."""
        driver = _driver(tmp_path, relaunch=_raise_boom)

        driver._relaunch(4)

        with pytest.raises(AssertionError, match="the hot restart driver failed"):
            driver.assert_all_restarts_happened()


def _raise_boom(_frozen_rollout_id: int | None) -> None:
    raise RuntimeError("boom")
