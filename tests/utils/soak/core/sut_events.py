import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter
from tests.utils.soak.core.events import SoakObservationEvent

from miles.utils.audit_utils.event_logger.models import Event

logger = logging.getLogger(__name__)


class SutFileKey(NamedTuple):
    name: str
    inode: int


SutEventCursor = dict[SutFileKey, int]

_adapter = TypeAdapter(list[Event])


@dataclass(frozen=True)
class SutEventBatch:
    events: list[Event]
    cursor: SutEventCursor


@dataclass(kw_only=True)
class SutEventFeed:
    directory: Path
    file_patterns: tuple[str, ...]
    event_types: tuple[type[Event], ...]
    cursor: SutEventCursor = field(default_factory=dict)

    async def attach(self, observation: SoakObservationEvent) -> SoakObservationEvent:
        try:
            batch = await self._observe()
        except Exception as error:
            logger.warning("Failed to observe system-under-test events", exc_info=True)
            return observation.model_copy(update={"errors": {**observation.errors, "sut_events": repr(error)}})
        self.cursor = batch.cursor
        return observation.model_copy(update={"new_sut_events": batch.events})

    async def _observe(self) -> SutEventBatch:
        return await asyncio.to_thread(self._read_events)

    def _read_events(self) -> SutEventBatch:
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)

        paths = sorted(path for pattern in self.file_patterns for path in self.directory.glob(pattern))

        payloads: list[dict] = []
        consumed: SutEventCursor = {}
        for path in paths:
            key = SutFileKey(name=path.name, inode=path.stat().st_ino)
            with path.open("rb") as stream:
                lines = stream.readlines()
            if lines and not lines[-1].endswith(b"\n"):
                lines.pop()
            already_read = self.cursor.get(key, 0)
            payloads += [
                json.loads(line) for line in lines[already_read if already_read <= len(lines) else 0 :] if line.strip()
            ]
            consumed[key] = len(lines)

        events = [event for event in _adapter.validate_python(payloads) if isinstance(event, self.event_types)]
        return SutEventBatch(events=events, cursor={**self.cursor, **consumed})
