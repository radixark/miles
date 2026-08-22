from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    RedoneFromScratch,
    assert_unsaved_run_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import compute_checkpoint_dir

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
SCHEDULE: tuple[ScheduledFreeze, ...] = (ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None),)


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
        if read_checkpoint_tracker_iteration(self.checkpoint_dir) is None:
            event_logger_checkpoint.discard(self.megatron_args)

    def assert_redone_from_scratch(
        self,
        *,
        records: list[HotRestartRecord] | None = None,
        schedule: tuple[ScheduledFreeze, ...] = SCHEDULE,
        num_rollouts: int = 6,
    ) -> RedoneFromScratch:
        return assert_unsaved_run_redone_from_scratch(
            dump_dir=str(self.dump_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            records=_records() if records is None else records,
            num_rollouts=num_rollouts,
            schedule=schedule,
        )


def _records() -> list[HotRestartRecord]:
    return [HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=1)]


def _run_restarted_before_save(tmp_path: Path) -> _Run:
    run = _Run(dump_dir=tmp_path)
    run.train(0, 1)
    run.take_over()
    run.train(0, 1, 2, 3)
    run.save(3)
    run.train(4, 5)
    return run


class TestAssertARunThatHadSavedNothingWasRedoneFromScratch:
    def test_a_take_over_of_the_frozen_run_that_threw_away_its_two_steps_passes(self, tmp_path):
        """The run was frozen after step 1 holding nothing, so exactly steps 0 and 1 are trained twice."""
        run = _run_restarted_before_save(tmp_path)

        redone = run.assert_redone_from_scratch()

        assert redone.frozen_rollout_id == 1
        assert redone.attempts_of_rollout_id == {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}

    def test_a_carried_over_step_fails(self, tmp_path):
        """A save before the freeze leaves a snapshot to restore, and the restored steps are not retrained ones."""
        run = _Run(dump_dir=tmp_path)
        run.train(0)
        run.save(0)
        run.train(1)
        run.take_over()
        run.train(1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="carried them over"):
            run.assert_redone_from_scratch()

    def test_a_record_that_had_a_checkpoint_at_trigger_time_fails(self, tmp_path):
        """The record is the only thing saying which of the two take-over paths this dump describes."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="had saved iteration"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=0, frozen_rollout_id=1)]
            )

    def test_a_take_over_handed_a_run_frozen_at_another_step_fails(self, tmp_path):
        """The freeze is what pins how much work the take-over throws away."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="never pinned where it landed"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=2)]
            )

    def test_more_than_one_restart_fails(self, tmp_path):
        """Every take-over after the first resumes from what the one before it made the run save."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="taken over once"):
            run.assert_redone_from_scratch(records=_records() * 2)

    def test_a_schedule_freezing_the_run_more_than_once_fails(self, tmp_path):
        """The second freeze would find a run holding a checkpoint of its own."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="taken over once"):
            run.assert_redone_from_scratch(
                schedule=(*SCHEDULE, ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3))
            )

    def test_a_schedule_freezing_the_run_on_a_checkpoint_fails(self, tmp_path):
        """That take-over resumes from a save, which is exactly what the checkpointed mode measures."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="already covers"):
            run.assert_redone_from_scratch(
                schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),),
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=2)],
            )

    def test_a_freeze_later_than_the_schedule_fails(self, tmp_path):
        """The log moved aside is what the frozen run had trained, so it ends where the freeze says."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2)
        run.take_over()
        run.train(0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="every step the frozen run had trained"):
            run.assert_redone_from_scratch()

    def test_a_restart_past_step_zero_fails(self, tmp_path):
        """Resuming at step 1 is a resume from a checkpoint, however it came about."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="not each of the 6 steps exactly once"):
            run.assert_redone_from_scratch()

    def test_a_second_discarded_log_fails(self, tmp_path):
        """One take-over throws away one run's worth of work, and a third attempt is a second take-over."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="instead of exactly one"):
            run.assert_redone_from_scratch()

    def test_a_step_logged_twice_fails(self, tmp_path):
        """This is the symptom the product fix removes: one log holding both attempts at a redone step."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 0, 1, 2, 3)
        run.save(3)
        run.train(4, 5)

        with pytest.raises(AssertionError, match="more than once"):
            run.assert_redone_from_scratch()

    def test_a_run_that_never_finished_every_step_fails(self, tmp_path):
        """A comparison over fewer steps than the run was asked for would quietly prove less."""
        run = _run_restarted_before_save(tmp_path)

        with pytest.raises(AssertionError, match="not each of the 7 steps exactly once"):
            run.assert_redone_from_scratch(num_rollouts=7)

    def test_a_run_that_never_saved_after_it_was_restarted_fails(self, tmp_path):
        """The point of restarting before the first save is to watch the save that follows it."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="nothing about saving"):
            run.assert_redone_from_scratch()

    def test_a_checkpoint_predating_the_freeze_fails(self, tmp_path):
        """A save the take-over threw away says nothing about the run saving once it had been restarted."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.take_over()
        run.train(0, 1, 2, 3, 4, 5)
        run.save(1)

        with pytest.raises(AssertionError, match="nothing here was saved after the take-over"):
            run.assert_redone_from_scratch()
