from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    RedoneSteps,
    assert_only_the_steps_after_a_checkpoint_were_redone,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import compute_checkpoint_dir

from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
SCHEDULE: tuple[ScheduledFreeze, ...] = (
    ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
    ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
)


def _write_finished_step(events_dir: Path, *, rollout_id: int) -> None:
    logger = EventLogger(log_dir=events_dir, file_name="main.jsonl", source=SimpleProcessIdentity(component="main"))
    logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {"train/grad_norm": 1.0}}, print_log=False)


@dataclass(frozen=True)
class _Run:
    dump_dir: Path

    @property
    def events_dir(self) -> Path:
        return self.dump_dir / EVENTS_DIRNAME

    @property
    def checkpoint_dir(self) -> Path:
        return Path(compute_checkpoint_dir(str(self.dump_dir)))

    @property
    def megatron_args(self) -> Namespace:
        return Namespace(
            save=str(self.checkpoint_dir),
            load=str(self.checkpoint_dir),
            save_debug_event_data=str(self.events_dir),
        )

    def train(self, *rollout_ids: int) -> None:
        for rollout_id in rollout_ids:
            _write_finished_step(self.events_dir, rollout_id=rollout_id)

    def save(self, iteration: int) -> None:
        event_logger_checkpoint.snapshot(self.megatron_args, iteration)
        (self.checkpoint_dir / TRACKER_FILENAME).write_text(str(iteration))

    def take_over(self) -> None:
        event_logger_checkpoint.restore(self.megatron_args)

    def assert_redone_steps(
        self,
        *,
        records: list[HotRestartRecord],
        schedule: Sequence[ScheduledFreeze] = SCHEDULE,
        num_rollouts: int = 6,
    ) -> RedoneSteps:
        return assert_only_the_steps_after_a_checkpoint_were_redone(
            dump_dir=str(self.dump_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            records=records,
            num_rollouts=num_rollouts,
            schedule=schedule,
        )


def _remove_tree(path: Path) -> None:
    for child in sorted(path.iterdir()):
        child.unlink() if child.is_file() else _remove_tree(child)
    path.rmdir()


def _records() -> list[HotRestartRecord]:
    return [
        HotRestartRecord(
            index=index, saved_iteration_at_trigger=one.saved_iteration, frozen_rollout_id=one.frozen_rollout_id
        )
        for index, one in enumerate(SCHEDULE)
    ]


def _run_restarted_twice(tmp_path: Path) -> _Run:
    run = _Run(dump_dir=tmp_path)
    run.train(0, 1)
    run.save(1)
    run.train(2)
    run.take_over()
    run.train(2, 3)
    run.save(3)
    run.train(4)
    run.take_over()
    run.train(4, 5)
    return run


class TestAssertOnlyTheStepsAfterACheckpointWereRedone:
    def test_two_take_overs_that_each_redid_their_own_window_pass(self, tmp_path):
        """Steps 2 and 4 were trained twice, every other step once, and no step three times."""
        run = _run_restarted_twice(tmp_path)

        redone = run.assert_redone_steps(records=_records())

        assert redone.resume_rollout_ids == (1, 3)
        assert redone.frozen_rollout_ids == (2, 4)
        assert redone.attempts_of_rollout_id == {0: 1, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1}

    def test_a_run_that_left_no_discarded_event_log_fails(self, tmp_path):
        """A take-over that never rolled the log back never resumed from a checkpoint either."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="leaves the log it rolled back"):
            run.assert_redone_steps(records=_records())

    def test_more_discarded_logs_than_the_driver_recorded_restarts_fails(self, tmp_path):
        """A take-over nobody triggered replaced this run, and no window explains what it redid."""
        run = _run_restarted_twice(tmp_path)

        with pytest.raises(AssertionError, match="leaves the log it rolled back"):
            run.assert_redone_steps(records=_records()[:1], schedule=SCHEDULE[:1])

    def test_a_take_over_handed_another_frozen_run_than_the_scenario_pinned_fails(self, tmp_path):
        """The schedule says where the run stands when a take-over is handed it, and drift says nothing."""
        run = _run_restarted_twice(tmp_path)
        drifted = _records()
        drifted[0] = HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=3)

        with pytest.raises(AssertionError, match="the schedule pins the"):
            run.assert_redone_steps(records=drifted)

    def test_a_take_over_triggered_against_a_run_holding_no_checkpoint_fails(self, tmp_path):
        """That is the other scenario, and the windows this one measures are the ones a checkpoint bounds."""
        run = _run_restarted_twice(tmp_path)
        without_checkpoint = _records()
        without_checkpoint[0] = HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=2)

        with pytest.raises(AssertionError, match="holding no checkpoint"):
            run.assert_redone_steps(records=without_checkpoint)

    def test_a_run_that_trained_past_the_step_it_was_frozen_at_fails(self, tmp_path):
        """A run that moved on was never frozen, and what its take-over cost is a race again."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.save(1)
        run.train(2, 3)
        run.take_over()
        run.train(2, 3, 4, 5)

        with pytest.raises(AssertionError, match="never pinned where a take-over landed"):
            run.assert_redone_steps(records=_records()[:1], schedule=SCHEDULE[:1])

    def test_a_take_over_that_resumed_from_another_checkpoint_than_the_pinned_one_fails(self, tmp_path):
        """The pinned save is the one the frozen run holds, so any other resume point is unexplained."""
        run = _Run(dump_dir=tmp_path)
        run.train(0)
        run.save(0)
        run.train(1, 2)
        run.take_over()
        run.train(1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="the logs say they resumed at"):
            run.assert_redone_steps(records=_records()[:1], schedule=SCHEDULE[:1])

    def test_a_take_over_that_trained_from_scratch_fails(self, tmp_path):
        """Resuming at step 0 instead of at the checkpoint redoes more than the pinned window."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.save(1)
        run.train(2)
        run.events_dir.rename(tmp_path / ".trash_20260818_000000_abcdef01")
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="the logs say they resumed at"):
            run.assert_redone_steps(records=_records()[:1], schedule=SCHEDULE[:1])

    def test_a_resume_point_no_checkpoint_of_this_run_holds_fails(self, tmp_path):
        """The tracker alone says an iteration; only the log beside it says the run really resumed from there."""
        run = _run_restarted_twice(tmp_path)
        for snapshot_dir in sorted(run.checkpoint_dir.glob("iter_*/debug_events")):
            _remove_tree(snapshot_dir)

        with pytest.raises(AssertionError, match="no event log snapshot"):
            run.assert_redone_steps(records=_records())

    def test_a_run_that_never_finished_every_step_fails(self, tmp_path):
        """A comparison over fewer steps than the run was asked for would quietly prove less."""
        run = _run_restarted_twice(tmp_path)

        with pytest.raises(AssertionError, match="exactly once"):
            run.assert_redone_steps(records=_records(), num_rollouts=7)

    def test_a_step_trained_a_third_time_fails(self, tmp_path):
        """Two take-overs sharing a window waste one step twice over, which no checkpoint explains."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.save(1)
        run.train(2)
        run.take_over()
        run.train(2, 3)
        run.take_over()
        run.train(2, 3, 4, 5)

        with pytest.raises(AssertionError, match="once or twice"):
            run.assert_redone_steps(
                records=[
                    HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),
                    HotRestartRecord(index=1, saved_iteration_at_trigger=1, frozen_rollout_id=3),
                ],
                schedule=(
                    ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
                    ScheduledFreeze(frozen_rollout_id=3, saved_iteration=1),
                ),
            )

    def test_one_log_describing_a_step_twice_fails(self, tmp_path):
        """A script that trained a step twice without any rollback is a run nothing rolled back at all."""
        run = _run_restarted_twice(tmp_path)
        run.train(5)

        with pytest.raises(AssertionError, match="more than once"):
            run.assert_redone_steps(records=_records())
