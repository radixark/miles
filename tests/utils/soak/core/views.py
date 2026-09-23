from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakEvent,
    SoakEvidenceArchivedEvent,
    SoakObservationEvent,
)
from tests.utils.soak.core.types import SoakTarget

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import Event, TrainGroupStepEndEvent, WeightUpdateResultEvent
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


def alive_targets_of_kind(observation: SoakObservationEvent, kind: str) -> list[SoakTarget]:
    return [target for target in observation.targets or [] if target.kind == kind and target.alive]


def quiescent_polls_of_type(events: list[SoakEvent], *, expected_count_of_kind: dict[str, int]) -> dict[str, int]:
    polls = dict.fromkeys(expected_count_of_kind, 0)
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            polls[event.request.target.kind] = 0
        elif isinstance(event, SoakObservationEvent) and event.targets is not None:
            for kind, expected_count in expected_count_of_kind.items():
                polled = [target for target in event.targets if target.kind == kind]
                settled = len(polled) == expected_count and all(target.alive for target in polled)
                polls[kind] = polls[kind] + 1 if settled else 0
    return polls


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


def compute_num_injections(events: list[SoakEvent], *, kind: str | None = None) -> int:
    return len(_applied_actions(events, kind=kind))


def compute_injection_times(events: list[SoakEvent], *, kind: str | None = None) -> list[datetime]:
    return [action.applied.timestamp for action in _applied_actions(events, kind=kind)]


def compute_successful_form_names(events: list[SoakEvent], *, kind: str) -> set[str]:
    return {action.requested.request.form_name for action in _applied_actions(events, kind=kind)}


def event_source(events: list[SoakEvent], *, name: str, fallback: Path) -> Path:
    for event in reversed(events):
        if isinstance(event, SoakEvidenceArchivedEvent):
            assert name not in event.missing_sources, f"Missing archived soak evidence: {name}"
            if name in event.sources:
                return event.sources[name]
    return fallback


def training_events_dir(events: list[SoakEvent], *, dump_dir: str | Path) -> Path:
    return event_source(events, name="training_events", fallback=Path(dump_dir) / EVENTS_DIRNAME)


def read_training_events(events: list[SoakEvent], *, dump_dir: str | Path) -> list[Event]:
    return read_events(training_events_dir(events, dump_dir=dump_dir))


def _applied_actions(events: list[SoakEvent], *, kind: str | None) -> list[SoakActionRecord]:
    return [
        action
        for action in project_actions(events).values()
        if action.applied is not None and (kind is None or action.requested.request.target.kind == kind)
    ]


# ================================ weight updates ==============================


class PublishedWeightUpdateKey(NamedTuple):
    trainer_model_id: str | None
    debug_trainer_load_state_timestamp: float
    weight_version: int
    debug_weight_update_id: str


def weight_update_results(events: Sequence[Event]) -> list[WeightUpdateResultEvent]:
    results = [event for event in events if isinstance(event, WeightUpdateResultEvent)]
    for result in results:
        updated, failed = set(result.updated_cell_ids), set(result.failed_cell_ids)
        assert len(updated) == len(
            result.updated_cell_ids
        ), f"Repeated updated engine: {result.debug_weight_update_id}"
        assert not updated & failed, f"Published engine is also reported failed: {result.debug_weight_update_id}"
        assert updated | failed == set(
            result.snapshot_cell_id_to_hashes
        ), f"Update omits assigned targets: {result.debug_weight_update_id}"
        assert all(
            result.snapshot_cell_id_to_hashes[cell_id] for cell_id in updated
        ), f"Updated engine lacks its incarnation: {result.debug_weight_update_id}"
        assert result.published_version == (
            result.candidate_version if updated else None
        ), f"Published version is inconsistent with the updated engines: {result.debug_weight_update_id}"
    return results


def published_weight_updates(events: Sequence[Event]) -> dict[PublishedWeightUpdateKey, WeightUpdateResultEvent]:
    published: dict[PublishedWeightUpdateKey, WeightUpdateResultEvent] = {}
    updates: set[tuple[str | None, str, str]] = set()
    for result in weight_update_results(events):
        if result.published_version is None:
            continue
        assert isinstance(result.source, TrainerControllerProcessIdentity), "Weight publication lacks trainer identity"
        assert (
            result.debug_trainer_load_state_timestamp and result.debug_weight_update_id
        ), "Publication lacks load-state timestamp or update identity"
        update = (result.source.model_id, result.debug_trainer_load_state_timestamp, result.debug_weight_update_id)
        assert update not in updates, f"Repeated weight update identity: {update}"
        updates.add(update)
        published[PublishedWeightUpdateKey(*update[:2], result.published_version, result.debug_weight_update_id)] = (
            result
        )
    return published
