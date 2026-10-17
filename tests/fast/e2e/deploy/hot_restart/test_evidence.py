from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import RELEASE, TRAINER, cluster_snapshot, pod_fact, workload_fact
from tests.utils.deploy.hot_restart.evidence import (
    CHECKPOINT_TRACKER_FILENAME,
    HotRestartEvidence,
    HotRestartRecord,
    RunProgress,
    read_discarded_event_dirs,
    read_finished_rollout_ids,
    read_finished_steps_once,
    read_last_saved_iteration,
    read_run_progress,
)

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


def _write_metric_event(events_dir: Path, *, rollout_id: int | None, metrics: dict[str, float]) -> None:
    logger = EventLogger(log_dir=events_dir, file_name="main.jsonl", source=SimpleProcessIdentity(component="main"))
    logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": metrics}, print_log=False)


def _write_tracker(checkpoint_dir: Path, *, relative: str, iteration: str) -> None:
    path = checkpoint_dir / relative / CHECKPOINT_TRACKER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{iteration}\n")


class TestReadLastSavedIteration:
    def test_a_run_that_never_saved_reports_nothing(self, tmp_path):
        """Restarting before the first checkpoint would resume a run from weights nothing wrote."""
        assert read_last_saved_iteration(tmp_path) is None

    def test_a_missing_checkpoint_directory_reports_nothing(self, tmp_path):
        """The directory only appears with the first save, and polling starts before that."""
        assert read_last_saved_iteration(tmp_path / "absent") is None

    def test_the_tracker_names_the_iteration_a_take_over_would_resume_from(self, tmp_path):
        """The saved iteration is the rollout id the trainer would reload, and the window starts after it."""
        _write_tracker(tmp_path, relative=".", iteration="3")

        assert read_last_saved_iteration(tmp_path) == 3

    def test_a_run_saving_several_trackers_is_refused(self, tmp_path):
        """One restart window cannot be read off several trainers that each save at their own pace."""
        _write_tracker(tmp_path, relative="trainers/solver", iteration="2")
        _write_tracker(tmp_path, relative="trainers/verifier", iteration="4")

        with pytest.raises(AssertionError, match="several policies"):
            read_last_saved_iteration(tmp_path)

    def test_a_half_written_tracker_is_not_read_as_a_save(self, tmp_path):
        """A save observed mid-write would open the gate on a checkpoint no trainer can load."""
        _write_tracker(tmp_path, relative=".", iteration="")

        assert read_last_saved_iteration(tmp_path) is None


class TestReadFinishedRolloutIds:
    def test_a_run_with_no_events_yet_reports_nothing(self, tmp_path):
        """Polling starts before the pods are even up."""
        assert read_finished_rollout_ids(tmp_path / "events") == []

    def test_only_a_metric_of_a_finished_train_step_counts(self, tmp_path):
        """Rollout-side metrics are logged before the trainer has moved any weights."""
        events_dir = tmp_path / "events"
        _write_metric_event(events_dir, rollout_id=0, metrics={"train/grad_norm": 1.5})
        _write_metric_event(events_dir, rollout_id=1, metrics={"rollout/response_length": 12.0})
        _write_metric_event(events_dir, rollout_id=None, metrics={"train/grad_norm": 2.0})

        assert read_finished_rollout_ids(events_dir) == [0]

    def test_the_ids_a_run_finished_are_reported_in_order_without_repeats(self, tmp_path):
        """A step logs several metric events, and the gate counts steps, not events."""
        events_dir = tmp_path / "events"
        for rollout_id in (2, 0, 2, 1):
            _write_metric_event(events_dir, rollout_id=rollout_id, metrics={"train/grad_norm": 1.0})

        assert read_finished_rollout_ids(events_dir) == [0, 1, 2]


class TestReadRunProgress:
    def test_progress_is_the_last_save_and_the_last_finished_step(self, tmp_path):
        """The gate is a function of exactly these two numbers, so they are read together."""
        _write_tracker(tmp_path / "ckpt", relative=".", iteration="1")
        _write_metric_event(tmp_path / "events", rollout_id=2, metrics={"train/grad_norm": 1.0})

        progress = read_run_progress(checkpoint_dir=tmp_path / "ckpt", events_dir=tmp_path / "events")

        assert progress == RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)


class TestHotRestartEvidence:
    def test_what_the_target_side_observed_survives_to_the_comparison(self, tmp_path):
        """The compare step runs as a subcommand of its own and cannot watch the run itself."""
        evidence = HotRestartEvidence(
            records=(HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),),
            snapshots=(
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]),
            ),
            release=RELEASE,
        )
        evidence.write(dump_dir=str(tmp_path))

        assert HotRestartEvidence.load(dump_dir=str(tmp_path)) == evidence

    def test_comparing_without_a_target_run_fails_loudly(self, tmp_path):
        """A missing file would otherwise read as a run that redid nothing."""
        with pytest.raises(AssertionError):
            HotRestartEvidence.load(dump_dir=str(tmp_path))


class TestReadFinishedStepsOnce:
    def test_each_finished_step_maps_to_the_one_event_that_logged_it(self, tmp_path: Path) -> None:
        """The redo checks compare steps by their logged event, so every finished step must be keyed once."""
        events_dir = tmp_path / "events"
        _write_metric_event(events_dir, rollout_id=1, metrics={"train/grad_norm": 1.0})
        _write_metric_event(events_dir, rollout_id=0, metrics={"train/grad_norm": 2.0})
        _write_metric_event(events_dir, rollout_id=2, metrics={"rollout/response_length": 3.0})

        steps = read_finished_steps_once(events_dir, what="the target log")

        assert list(steps) == [0, 1]
        assert '"train/grad_norm":2.0' in steps[0].replace(" ", "")

    def test_a_step_logged_twice_is_refused_naming_the_log_and_the_step(self, tmp_path: Path) -> None:
        """A take-over that failed to roll the log back would otherwise pass with two copies of one step."""
        events_dir = tmp_path / "events"
        for rollout_id in (0, 1, 1):
            _write_metric_event(events_dir, rollout_id=rollout_id, metrics={"train/grad_norm": 1.0})

        with pytest.raises(AssertionError, match=r"the target log describes the step\(s\) \{1: 2\}"):
            read_finished_steps_once(events_dir, what="the target log")


class TestReadDiscardedEventDirs:
    def test_only_the_rolled_aside_logs_are_listed_in_name_order(self, tmp_path: Path) -> None:
        """The checks walk rolled-aside logs oldest first, and the live log must never be read as discarded."""
        for name in (".trash_20260102-000000", "events", ".trash_20260101-000000", "trash_unmarked"):
            (tmp_path / name).mkdir()

        assert read_discarded_event_dirs(str(tmp_path)) == [
            tmp_path / ".trash_20260101-000000",
            tmp_path / ".trash_20260102-000000",
        ]
