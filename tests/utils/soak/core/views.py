from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.models import Event, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


@dataclass(frozen=True)
class SoakActionRecord:
    requested: SoakActionRequestedEvent
    applied: SoakActionAppliedEvent | None = None
    result: SoakActionResultEvent | None = None


def project_actions(events: list[SoakEvent]) -> dict[str, SoakActionRecord]:
    requested = {event.request.request_id: event for event in events if isinstance(event, SoakActionRequestedEvent)}
    applied = {event.request_id: event for event in events if isinstance(event, SoakActionAppliedEvent)}
    results = {event.request_id: event for event in events if isinstance(event, SoakActionResultEvent)}
    return {
        request_id: SoakActionRecord(requested=event, applied=applied.get(request_id), result=results.get(request_id))
        for request_id, event in requested.items()
    }


# ================================= scheduling =================================


def admission_closed(events: list[SoakEvent]) -> SoakAdmissionClosedEvent | None:
    return next((event for event in events if isinstance(event, SoakAdmissionClosedEvent)), None)


def tail_started_at(events: list[SoakEvent]) -> datetime:
    closed = admission_closed(events)
    assert closed is not None, "Soak injection admission never closed"
    return max(
        [
            closed.timestamp,
            *(action.applied.timestamp for action in project_actions(events).values() if action.applied),
        ]
    )


def latest_observation(events: list[SoakEvent]) -> SoakObservationEvent | None:
    return next((event for event in reversed(events) if isinstance(event, SoakObservationEvent)), None)


# ================================ sut progress ================================


def sut_events(events: list[SoakEvent]) -> list[Event]:
    return [
        event
        for observation in events
        if isinstance(observation, SoakObservationEvent)
        for event in observation.new_sut_events
    ]


def trainer_step_ends(events: list[SoakEvent]) -> list[TrainGroupStepEndEvent]:
    return [
        event
        for event in sut_events(events)
        if isinstance(event, TrainGroupStepEndEvent)
        and isinstance(event.source, TrainerControllerProcessIdentity)
        and event.source.trainer_id == ACTOR_ROLE
    ]


def is_normal_step(step: TrainGroupStepEndEvent) -> bool:
    return any(
        isinstance(outcomes, list) and TrainStepOutcome.NORMAL in outcomes for outcomes in step.cell_outcomes.values()
    )


# ================================== injections ================================


def event_source(events: list[SoakEvent], *, name: str, fallback: Path) -> Path:
    raise NotImplementedError
