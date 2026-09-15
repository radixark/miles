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
    assert len(discarded_dirs) == 1, (
        f"the take-over of a run holding no checkpoint starts it over, and the log of what it threw away is moved "
        f"aside; {[one.name for one in discarded_dirs]} was left behind instead of exactly one"
    )
    [discarded_dir] = discarded_dirs

    discarded_log = _read_steps_written_once(discarded_dir, what=discarded_dir.name)
    assert sorted(discarded_log) == list(range(scheduled.frozen_rollout_id + 1)), (
        f"the run was frozen after step {scheduled.frozen_rollout_id}, and the log moved aside describes "
        f"{sorted(discarded_log)}: what the take-over threw away is every step the frozen run had trained"
    )

    surviving_log = _read_steps_written_once(Path(dump_dir) / EVENTS_DIRNAME, what="the surviving event log")
    assert sorted(surviving_log) == list(range(num_rollouts)), (
        f"the surviving event log describes {sorted(surviving_log)}, not each of the {num_rollouts} steps exactly "
        f"once: a take-over of a run holding no checkpoint restarts it at step 0 and it trains to the end from there"
    )

    carried = sorted(one for one, event in discarded_log.items() if surviving_log.get(one) == event)
    assert not carried, (
        f"the steps {carried} appear in the log the take-over moved aside and again, unchanged, in the log that "
        f"replaced it: the run carried them over instead of training them again from the reference weights"
    )

    logs = [discarded_log, surviving_log]
    attempts = {
        rollout_id: len({log[rollout_id] for log in logs if rollout_id in log}) for rollout_id in range(num_rollouts)
    }
    expected = compute_expected_attempts(num_rollouts=num_rollouts, schedule=schedule)
    assert attempts == expected, (
        f"the run was frozen after step {scheduled.frozen_rollout_id} holding no checkpoint, so the take-over threw "
        f"away exactly the steps 0..{scheduled.frozen_rollout_id}: every step is written {expected} time(s) across "
        f"the two logs, and this run left {attempts}"
    )

    _assert_run_saved_after_restart(
        checkpoint_dir=checkpoint_dir, dump_dir=dump_dir, frozen_rollout_id=scheduled.frozen_rollout_id
    )

    return RedoneFromScratch(frozen_rollout_id=scheduled.frozen_rollout_id, attempts_of_rollout_id=attempts)


def _read_steps_written_once(events_dir: Path, *, what: str) -> dict[int, str]:
    logged = read_step_events(events_dir)
    repeated = {rollout_id: len(events) for rollout_id, events in logged.items() if len(events) != 1}
    assert not repeated, (
        f"{what} describes the step(s) {repeated} more than once; a run that starts over writes what it retrains "
        f"into a log of its own, so no one log carries a step twice"
    )
    return {rollout_id: events[0] for rollout_id, events in logged.items()}


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


def _assert_run_saved_after_restart(*, checkpoint_dir: str, dump_dir: str, frozen_rollout_id: int) -> None:
    assert (saved := read_last_saved_iteration(Path(checkpoint_dir))) is not None, (
        f"{checkpoint_dir} holds no checkpoint even after the run ended, so this run says nothing about saving "
        f"once a take-over has restarted it"
    )
    assert saved > frozen_rollout_id, (
        f"{checkpoint_dir} holds iteration {saved}, which the run had already trained when it was frozen after "
        f"step {frozen_rollout_id}: nothing here was saved after the take-over restarted it"
    )
    snapshot_dirs = read_checkpoint_snapshot_dirs(checkpoint_dir)
    assert snapshot_dirs, (
        f"{checkpoint_dir} saved iteration {saved} without the copy of {dump_dir}/{EVENTS_DIRNAME} every save "
        f"writes beside it, so the save after the restart was not a whole one"
    )
