from datetime import datetime
from pathlib import Path

from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import compute_injection_times, event_source
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.metrics import read_rollout_completion_times

MIN_FAULT_PROGRESS_WINDOWS: int = 2


def assert_faults_span_progress_windows(events: list[SoakEvent], *, dump_dir: str) -> None:
    source = event_source(events, name="training_events", fallback=Path(dump_dir) / EVENTS_DIRNAME)
    progress_windows = _compute_fault_progress_windows(
        injected_at=compute_injection_times(events, kind=ROLLOUT_CELL_TYPE),
        rollout_completions=read_rollout_completion_times(str(source.parent)),
    )

    assert len(progress_windows) >= MIN_FAULT_PROGRESS_WINDOWS, (
        f"Fault effects occupy only {sorted(progress_windows)} progress windows; "
        f"expected at least {MIN_FAULT_PROGRESS_WINDOWS} windows separated by completed rollouts"
    )
    print(f"Fault effects span progress windows {sorted(progress_windows)}")


def _compute_fault_progress_windows(
    *, injected_at: list[datetime], rollout_completions: list[tuple[int, datetime]]
) -> set[int]:
    return {
        max((rollout_id for rollout_id, finished_at in rollout_completions if finished_at <= at), default=-1) + 1
        for at in injected_at
    }
