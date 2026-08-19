from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    compute_expected_attempts,
    read_checkpoint_snapshot_dirs,
    read_discarded_event_dirs,
    read_step_events,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord, read_last_saved_iteration

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME


@dataclass(frozen=True)
class RedoneFromScratch:
    frozen_rollout_id: int
    attempts_of_rollout_id: dict[int, int]


def assert_unsaved_run_redone_from_scratch(
    *,
    dump_dir: str,
    checkpoint_dir: str,
    records: Sequence[HotRestartRecord],
    num_rollouts: int,
    schedule: Sequence[ScheduledFreeze],
) -> RedoneFromScratch:
    scheduled = _read_only_freeze_without_prior_save(schedule)
    record = _read_only_record_without_save(records)
    assert record.frozen_rollout_id == scheduled.frozen_rollout_id, (
        f"this scenario freezes the run after step {scheduled.frozen_rollout_id}, and the take-over was handed a "
        f"run standing at step {record.frozen_rollout_id}: the freeze never pinned where it landed"
    )

    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert not discarded_dirs, (
        f"a run that had saved nothing resumes from --ref-load, which holds no snapshot to restore, so no log is "
        f"moved aside; {[one.name for one in discarded_dirs]} was left behind, so this run resumed from a "
        f"checkpoint after all"
    )

    attempts = {
        rollout_id: len(logged) for rollout_id, logged in read_step_events(Path(dump_dir) / EVENTS_DIRNAME).items()
    }
    assert sorted(attempts) == list(range(num_rollouts)), (
        f"the run was asked for {num_rollouts} steps and its one event log describes {sorted(attempts)}; a "
        f"take-over of a run holding no checkpoint restarts it at step 0 and it trains to the end from there"
    )

    expected = compute_expected_attempts(num_rollouts=num_rollouts, schedule=schedule)
    assert attempts == expected, (
        f"the run was frozen after step {scheduled.frozen_rollout_id} holding no checkpoint, so the take-over threw "
        f"away exactly the steps 0..{scheduled.frozen_rollout_id}: every step is written {expected} time(s), and "
        f"this run left {attempts}"
    )

    _assert_run_saved_after_restart(checkpoint_dir=checkpoint_dir, dump_dir=dump_dir)

    return RedoneFromScratch(frozen_rollout_id=scheduled.frozen_rollout_id, attempts_of_rollout_id=attempts)


def _read_only_freeze_without_prior_save(
    schedule: Sequence[ScheduledFreeze],
) -> ScheduledFreeze:
    assert len(schedule) == 1, (
        f"a run holds no checkpoint only until its first save, so it is taken over once; this schedule pins "
        f"{len(schedule)} freeze(s), and every take-over after the first resumes from something it had written"
    )
    [scheduled] = schedule
    assert scheduled.saved_iteration is None, (
        f"this schedule freezes the run holding the checkpoint of iteration {scheduled.saved_iteration}, which is "
        f"the take-over the checkpointed mode already covers"
    )
    return scheduled


def _read_only_record_without_save(records: Sequence[HotRestartRecord]) -> HotRestartRecord:
    assert len(records) == 1, (
        f"a run holds no checkpoint only until its first save, so it is taken over once; {len(records)} restart(s) "
        f"were recorded, and every one after the first resumed from something it had written"
    )
    [record] = records
    assert record.saved_iteration_at_trigger is None, (
        f"restart {record.index} fired against a run that had saved iteration "
        f"{record.saved_iteration_at_trigger}, so the take-over this dump describes had a checkpoint to resume from"
    )
    return record


def _assert_run_saved_after_restart(*, checkpoint_dir: str, dump_dir: str) -> None:
    assert (saved := read_last_saved_iteration(Path(checkpoint_dir))) is not None, (
        f"{checkpoint_dir} holds no checkpoint even after the run ended, so this run says nothing about saving "
        f"once a take-over has restarted it"
    )
    snapshot_dirs = read_checkpoint_snapshot_dirs(checkpoint_dir)
    assert snapshot_dirs, (
        f"{checkpoint_dir} saved iteration {saved} without the copy of {dump_dir}/{EVENTS_DIRNAME} every save "
        f"writes beside it, so the save after the restart was not a whole one"
    )
