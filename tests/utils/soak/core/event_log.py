import os
from pathlib import Path

from tests.utils.soak.core.events import SoakCollectionClosedEvent, SoakEvent, StoredEvent


class EventLog:
    def __init__(self) -> None:
        self._events: list[SoakEvent] = []
        self._path: Path | None = None

    def persist_to(self, path: Path) -> None:
        assert self._path is None and not self._events, "Configure evidence persistence before recording events"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x"):
            pass
        self._path = path

    @property
    def events(self) -> list[SoakEvent]:
        return list(self._events)

    @property
    def path(self) -> Path | None:
        return self._path

    def append(self, event: SoakEvent) -> None:
        assert not self._events or not isinstance(
            self._events[-1], SoakCollectionClosedEvent
        ), "Soak evidence is closed"
        snapshot = type(event).model_validate(event.model_dump(mode="json"))
        if self._path is not None:
            stored = StoredEvent(
                sequence=len(self._events),
                event_type=type(snapshot).__name__,
                event=snapshot.model_dump(mode="json"),
            )
            with self._path.open("r+b") as stream:
                stream.seek(0, os.SEEK_END)
                stream.write((stored.model_dump_json() + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
        self._events.append(snapshot)
