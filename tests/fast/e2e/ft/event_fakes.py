from pathlib import Path
from typing import Any

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    EventBase,
    FaultHookEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, FaultHookRequest, FaultHookStatus

_Event = tuple[type[EventBase], dict[str, Any]]


def _write_events(events_dir: Path, events: list[_Event]) -> None:
    event_logger = EventLogger(log_dir=events_dir, source=SimpleProcessIdentity(component="main"))
    for event_cls, partial in events:
        event_logger.log(event_cls, partial, print_log=False)


def _reconfigure(*, rollout_id: int, healed: list[int], alive: list[int], src: int | None = 0) -> _Event:
    return CellReconfigureEvent, dict(
        rollout_id=rollout_id,
        quorum_id=rollout_id,
        src_cell_index=src,
        healed_cell_indices=healed,
        alive_cell_indices_after=alive,
    )


def _hook(request_id: str, status: FaultHookStatus) -> _Event:
    request = FaultHookRequest(request_id=request_id, action=KillProcessAction())
    return FaultHookEvent, dict(record=FaultHookRecord(request=request, status=status, set_at=0.0, changed_at=1.0))


def _update(*, rollout_id: int | None, published_version: int | None = 5, failed: list[str] | None = None) -> _Event:
    return WeightUpdateResultEvent, dict(
        debug_weight_update_id=f"update-{rollout_id}",
        debug_trainer_load_state_timestamp=0.0,
        rollout_id=rollout_id,
        candidate_version=5,
        published_version=published_version,
        snapshot_cell_id_to_hashes={},
        updated_cell_ids=[],
        failed_cell_ids=failed or [],
    )
