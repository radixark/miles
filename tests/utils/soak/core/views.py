from dataclasses import dataclass
from pathlib import Path

from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakEvent,
)


@dataclass(frozen=True)
class SoakActionRecord:
    requested: SoakActionRequestedEvent
    applied: SoakActionAppliedEvent | None = None
    result: SoakActionResultEvent | None = None


# ================================== injections ================================


def event_source(events: list[SoakEvent], *, name: str, fallback: Path) -> Path:
    raise NotImplementedError
