from pathlib import Path
from typing import Any

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, EventBase
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

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
