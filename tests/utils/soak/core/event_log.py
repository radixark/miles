from pathlib import Path

from tests.utils.soak.core.events import SoakEvent


class EventLog:
    def __init__(self, path: Path) -> None:
        self._events: list[SoakEvent] = []
        self._path = path

    @property
    def events(self) -> list[SoakEvent]:
        return list(self._events)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: SoakEvent) -> None:
        raise NotImplementedError
