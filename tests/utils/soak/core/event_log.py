import os
from pathlib import Path

from tests.utils.soak.core.events import SoakCollectionClosedEvent, SoakEvent, StoredEvent


class EventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x"):
            pass

        self._events: list[SoakEvent] = []
        self._path = path

    @property
    def events(self) -> list[SoakEvent]:
        return list(self._events)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: SoakEvent) -> None:
        assert not self._events or not isinstance(
            self._events[-1], SoakCollectionClosedEvent
        ), "Soak evidence is closed"
        snapshot = type(event).model_validate(event.model_dump(mode="json"))
        stored = StoredEvent(sequence=len(self._events), event=snapshot)
        with self._path.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            stream.write((stored.model_dump_json() + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        self._events.append(snapshot)
