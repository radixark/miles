import threading
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import driver as driver_module
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    HOT_RESTART_ARG,
    HotRestartDriver,
    ScheduledFreeze,
    compute_freeze_plan,
    relaunch_with_hot_restart,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, RunProgress
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.fast.e2e.deploy.hot_restart.cluster_facts import NAMESPACE, RELEASE

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.ft_test_actions import SLEEP_FOREVER_AT_END_ACTION, FTTestAction
from miles.utils.workers.types import HotRestartComponent


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


class TestHotRestartArg:
    def test_the_flag_names_both_components_a_take_over_replaces(self):
        """A hot restart replaces the orchestration script and the rollout executor together or not at all."""
        assert sorted(HOT_RESTART_ARG.split(",")) == sorted(one.value for one in HotRestartComponent)


class TestRelaunchWithHotRestart:
    def test_a_relaunch_that_would_install_another_release_is_refused(self):
        """A relaunch building its own config gets a run id of its own and leaves the run behind."""
        mode = FTTestMode(
            model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
        )

        with pytest.raises(AssertionError, match="already up"):
            relaunch_with_hot_restart(
                train_args="",
                mode=mode,
                config=ExecuteTrainConfig(run_id="demo"),
                installed_release="miles-run-someone-else-all",
            )


def _driver(tmp_path: Path, **overrides: Any) -> HotRestartDriver:
    kwargs: dict[str, Any] = dict(
        relaunch=lambda _frozen_rollout_id: None,
        checkpoint_dir=tmp_path / "checkpoints",
        events_dir=tmp_path / "events",
        release=RELEASE,
        namespace=NAMESPACE,
        trainer_id="actor",
        freeze_plan_path=tmp_path / "hot_restart" / "target_freeze_plan.json",
        schedule=_SCHEDULE,
        tracker_settle_interval_seconds=0.0,
        poll_interval_seconds=0.0,
    )
    kwargs.update(overrides)
    return HotRestartDriver(**kwargs)


def _join_relaunches(driver: HotRestartDriver) -> None:
    for thread in driver._relaunch_threads:
        thread.join(timeout=30.0)


def _install_frozen_at(monkeypatch, rollout_id: int | None) -> None:
    monkeypatch.setattr(driver_module, "read_frozen_rollout_id", lambda _path: rollout_id)


def _install_progress(monkeypatch, reported: list[RunProgress]) -> None:
    remaining = list(reported)
    latest: list[RunProgress] = []

    def read(**_kwargs) -> RunProgress:
        latest.append(remaining.pop(0) if remaining else reported[-1])
        return latest[-1]

    monkeypatch.setattr(driver_module, "read_run_progress", read)
    monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
    monkeypatch.setattr(
        driver_module,
        "read_frozen_rollout_id",
        lambda _path: latest[-1].last_finished_rollout_id if latest else None,
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
    def test_a_run_that_has_finished_nothing_is_not_frozen_yet(self, tmp_path, monkeypatch):
        """Polling starts before the pods are even up."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=None)
        )

    def test_a_run_short_of_the_pinned_step_is_not_frozen_yet(self, tmp_path, monkeypatch):
        """Taking a run over one step early would resume it from another checkpoint than the pinned one."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=1)
        )

    def test_a_run_standing_at_the_pinned_step_that_has_not_parked_is_not_frozen_yet(self, tmp_path, monkeypatch):
        """The metric event is written before the sleep, so the step being over is not the run being parked."""
        _install_frozen_at(monkeypatch, None)

        assert not _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )

    def test_the_run_that_wrote_the_sentinel_at_the_pinned_step_is_the_frozen_one(self, tmp_path, monkeypatch):
        """The run writes it from inside its sleep loop, so it can no longer read a rewritten plan."""
        _install_frozen_at(monkeypatch, 2)

        assert _driver(tmp_path)._stands_frozen_at(
            _CHECKPOINTED, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )

    def test_a_sentinel_left_by_the_previous_take_over_is_not_read_as_this_one(self, tmp_path, monkeypatch):
        """Each freeze overwrites it, so the value only matches once the new run has parked."""
        _install_frozen_at(monkeypatch, 2)

        assert not _driver(tmp_path)._stands_frozen_at(
            ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
            progress=RunProgress(last_saved_iteration=3, last_finished_rollout_id=3),
        )

    def test_a_run_that_trained_past_the_pinned_step_fails_instead_of_firing(self, tmp_path, monkeypatch):
        """The freeze is what makes the take-over exact, and a run that moved on was never frozen."""
        _install_frozen_at(monkeypatch, None)

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
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
        monkeypatch.setattr(driver_module, "read_frozen_rollout_id", lambda _path: 2 if len(reads) > 1 else None)

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


class TestWatchingPastTheLastTakeOver:
    def test_the_run_is_still_sampled_after_the_last_take_over_was_triggered(self, tmp_path, monkeypatch):
        """The last relaunch is when its pods are replaced, and nothing else would ever look at them."""
        stop_event = threading.Event()
        driver = _driver(
            tmp_path, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),), poll_interval_seconds=0.0
        )
        _install_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)])

        observed_after: list[int] = []

        def observe(_self) -> None:
            observed_after.append(len(driver.records))
            if len(observed_after) >= 6:
                stop_event.set()

        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", observe)

        driver._drive(stop_event)
        _join_relaunches(driver)

        assert len(driver.records) == 1
        assert observed_after.count(1) >= 2, "the run has to keep being sampled once the schedule is spent"


class TestWhenTheDriverCannotReadTheRun:
    def test_a_read_that_never_answers_still_runs_out_of_time(self, tmp_path, monkeypatch):
        """The deadline used to be checked only on a successful read, so this hung to the ci limit."""
        driver = _driver(tmp_path, freeze_timeout_seconds=0.0)
        monkeypatch.setattr(driver_module, "read_run_progress", _refuse_to_read)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="waited 0.0s"):
            driver.assert_all_restarts_happened()

    def test_a_read_that_keeps_failing_is_reported_rather_than_retried_forever(self, tmp_path, monkeypatch):
        """A dump directory nobody can read is not a run that has not frozen yet."""
        driver = _driver(tmp_path, freeze_timeout_seconds=1_000_000.0, read_failure_limit=3)
        monkeypatch.setattr(driver_module, "read_run_progress", _refuse_to_read)
        monkeypatch.setattr(driver_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)

        driver._drive(threading.Event())

        with pytest.raises(AssertionError, match="3 time\\(s\\) in a row"):
            driver.assert_all_restarts_happened()


def _refuse_to_read(**_kwargs):
    raise OSError("the dump directory did not answer")


class TestWhenTheCheckpointTrackerLagsTheFreeze:
    def test_a_tracker_that_catches_up_within_a_few_reads_is_accepted(self, tmp_path, monkeypatch):
        """The tracker is written by the save itself, a moment after the step it belongs to ends."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.HotRestartDriver, "_reread_saved_iteration", lambda _self: 1)

        record = driver._compute_record(
            index=0,
            scheduled=_CHECKPOINTED,
            progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=2),
        )

        assert record.saved_iteration_at_trigger == 1

    def test_a_tracker_that_never_catches_up_still_fails(self, tmp_path, monkeypatch):
        """Re-reading is for a save in flight, not for a run saving at another pace altogether."""
        driver = _driver(tmp_path)
        monkeypatch.setattr(driver_module.HotRestartDriver, "_reread_saved_iteration", lambda _self: 5)

        with pytest.raises(AssertionError, match="save cadence"):
            driver._compute_record(
                index=0,
                scheduled=_CHECKPOINTED,
                progress=RunProgress(last_saved_iteration=5, last_finished_rollout_id=2),
            )
