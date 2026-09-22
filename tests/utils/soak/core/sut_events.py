import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter
from tests.utils.soak.core.events import SoakObservationEvent

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, Event, TrainGroupStepEndEvent

logger = logging.getLogger(__name__)

SUT_EVENT_TYPES: tuple[type, ...] = (TrainGroupStepEndEvent, CellReconfigureEvent)


class SutFileKey(NamedTuple):
    name: str
    inode: int


SutEventCursor = dict[SutFileKey, int]

_adapter = TypeAdapter(list[Event])


@dataclass(kw_only=True)
class SutEventFeed:
    directory: Path
    cursor: SutEventCursor = field(default_factory=dict)

    async def attach(self, observation: SoakObservationEvent) -> SoakObservationEvent:
        try:
            batch = await observe_sut_events(self.directory, cursor=self.cursor)
        except Exception as error:
            logger.warning("Failed to observe system-under-test events", exc_info=True)
            return observation.model_copy(update={"errors": {**observation.errors, "sut_events": repr(error)}})
        self.cursor = batch.cursor
        return observation.model_copy(update={"new_sut_events": batch.events})


@dataclass(frozen=True)
class SutEventBatch:
    events: list[Event]
    cursor: SutEventCursor


async def observe_sut_events(directory: Path, *, cursor: SutEventCursor) -> SutEventBatch:
    return await asyncio.to_thread(_read_events, directory, cursor=cursor)


def _read_events(directory: Path, *, cursor: SutEventCursor) -> SutEventBatch:
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    selected = {event_type.model_fields["type"].default for event_type in SUT_EVENT_TYPES}
    paths = sorted([*directory.glob("trainer_controller_*.jsonl"), *directory.glob("rollout_executor.jsonl")])

    payloads: list[dict] = []
    consumed: SutEventCursor = {}
    for path in paths:
        key = SutFileKey(name=path.name, inode=path.stat().st_ino)
        with path.open("rb") as stream:
            lines = stream.readlines()
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
        already_read = cursor.get(key, 0)
        for line in lines[already_read if already_read <= len(lines) else 0 :]:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload["type"] in selected:
                payloads.append(payload)
        consumed[key] = len(lines)

    return SutEventBatch(events=_adapter.validate_python(payloads), cursor={**cursor, **consumed})
