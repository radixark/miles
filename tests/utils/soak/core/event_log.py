from pathlib import Path

from tests.utils.soak.core.events import SoakEvent


class EventLog:
    def __init__(self) -> None:
        self._events: list[SoakEvent] = []
        self._path: Path | None = None

    def persist_to(self, path: Path) -> None:
        raise NotImplementedError

    @property
    def events(self) -> list[SoakEvent]:
        return list(self._events)

    @property
    def path(self) -> Path | None:
        return self._path

    def append(self, event: SoakEvent) -> None:
        raise NotImplementedError
