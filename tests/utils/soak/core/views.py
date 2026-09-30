from dataclasses import dataclass
from pathlib import Path

from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
)

from miles.utils.audit_utils.event_logger.models import Event


@dataclass(frozen=True)
class SoakActionRecord:
    requested: SoakActionRequestedEvent
    applied: SoakActionAppliedEvent | None = None
    result: SoakActionResultEvent | None = None


# ================================= scheduling =================================


def admission_closed(events: list[SoakEvent]) -> SoakAdmissionClosedEvent | None:
    return next((event for event in events if isinstance(event, SoakAdmissionClosedEvent)), None)


# ================================ sut progress ================================


def sut_events(events: list[SoakEvent]) -> list[Event]:
    return [
        event
        for observation in events
        if isinstance(observation, SoakObservationEvent)
        for event in observation.new_sut_events
    ]


# ================================== injections ================================


def event_source(events: list[SoakEvent], *, name: str, fallback: Path) -> Path:
    raise NotImplementedError
