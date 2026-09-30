import re
from collections.abc import Sequence
from pathlib import Path

from tests.utils.deploy.hot_restart.evidence import (
    HotRestartRecord,
    read_discarded_event_dirs,
    read_finished_steps_once,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.deploy.actions.hot_restart import HOT_RESTART_FORM_NAME, saved_iteration_after

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME

SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 2
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1


def assert_checkpoints_advanced_between_takeovers(events: list[SoakEvent]) -> None:
    previous_saved_iteration = -1
    count = 0
    for request_id, action in project_actions(events).items():
        if action.requested.request.form_name != HOT_RESTART_FORM_NAME or action.applied is None:
            continue
        target = action.requested.request.target
        assert target.saved_iteration is not None and target.saved_iteration > previous_saved_iteration, (
            f"Takeover {request_id} lacks a new checkpoint after the preceding takeover: "
            f"saved={target.saved_iteration}, previous={previous_saved_iteration}"
        )
        previous_saved_iteration = saved_iteration_after(action)
        count += 1
    assert count >= MIN_HOT_RESTARTS, f"Expected at least {MIN_HOT_RESTARTS} applied takeovers, got {count}"


def assert_take_over_loss_within_save_interval(records: Sequence[HotRestartRecord]) -> None:
    for record in records:
        resumed_from = -1 if record.saved_iteration_at_trigger is None else record.saved_iteration_at_trigger
        redone = record.frozen_rollout_id - resumed_from

        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} was drawn against a run standing at step {record.frozen_rollout_id} holding "
            f"iteration {record.saved_iteration_at_trigger}, so it threw away {redone} step(s); a take-over resumes "
            f"from the last checkpoint and a run saving every {SAVE_INTERVAL} step(s) cannot owe more than "
            f"{MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def assert_take_overs_resumed_within_save_interval(dump_dir: str, *, records: Sequence[HotRestartRecord]) -> None:
    logs = _read_replaced_logs(dump_dir, num_take_overs=len(records))

    for record, log, later_log in zip(records, logs[:-1], logs[1:], strict=True):
        frozen_rollout_id = max(log, default=-1)
        survived = sorted(rollout_id for rollout_id, event in log.items() if later_log.get(rollout_id) == event)
        resumed_from = max(survived, default=-1)

        assert survived == list(
            range(resumed_from + 1)
        ), f"take-over {record.index} carried the steps {survived} over, expected {list(range(resumed_from + 1))}"
        redone = frozen_rollout_id - resumed_from
        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} replaced a log that had reached step {frozen_rollout_id} and resumed at "
            f"step {resumed_from}, so it redid {redone} step(s), more than {MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def _read_replaced_logs(dump_dir: str, *, num_take_overs: int) -> list[dict[int, str]]:
    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert len(discarded_dirs) == num_take_overs, (
        f"every take-over rolls the log it replaced aside, but {num_take_overs} of them left "
        f"{[one.name for one in discarded_dirs]} under {dump_dir}"
    )

    rolled_aside_at = [_read_log_rollaside_times(one) for one in discarded_dirs]
    assert (
        sorted(set(rolled_aside_at)) == rolled_aside_at
    ), f"the take-overs under {dump_dir} rolled their logs aside at {rolled_aside_at}, two in the same second"

    replaced = [read_finished_steps_once(one, what=str(one)) for one in discarded_dirs]
    surviving = read_finished_steps_once(Path(dump_dir) / EVENTS_DIRNAME, what="the surviving event log")
    return [*replaced, surviving]


def _read_log_rollaside_times(events_dir: Path) -> str:
    matched = re.fullmatch(r"\.trash_(\d{8}_\d{6})_[0-9a-f]+", events_dir.name)
    assert matched is not None, f"{events_dir.name} does not name the moment the log was rolled aside"
    return matched.group(1)
