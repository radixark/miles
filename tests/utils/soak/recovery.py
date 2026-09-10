from dataclasses import dataclass, field
from datetime import datetime

from tests.utils.soak.batch import expand_fault_batches
from tests.utils.soak.state import (
    Event,
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakObservation,
    cell_is_alive,
    cell_type_of,
)

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.workers.naming import parse_cell_id


@dataclass
class RecoveryEpisode:
    cell_id: str
    cell_type: str
    last_requested_at: datetime
    last_applied_at: datetime
    request_ids: list[str] = field(default_factory=list)
    harmed_incarnations: set[str] = field(default_factory=set)
    recovered_incarnation: str | None = None
    recovered_at: datetime | None = None


def compute_recovery_episodes(
    events: list[Event],
    *,
    reconfigurations: list[CellReconfigureEvent] | None = None,
) -> list[RecoveryEpisode]:
    events = expand_fault_batches(events)
    applied = _index_applied_actions(events)
    episodes: list[RecoveryEpisode] = []
    pending: dict[str, RecoveryEpisode] = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            request = event.request
            if request.request_id not in applied or not request.harms_cell or not isinstance(request.target, dict):
                continue
            name = request.target["metadata"]["name"]
            incarnation = request.target["status"]["workers_hash"]
            if not incarnation:
                raise ValueError("Recovery evidence requires a nonempty target incarnation")
            effect = applied[request.request_id]
            if name not in pending:
                episode = RecoveryEpisode(
                    cell_id=name,
                    cell_type=cell_type_of(request.target),
                    last_requested_at=event.timestamp,
                    last_applied_at=effect.timestamp,
                )
                episodes.append(episode)
                pending[name] = episode
            episode = pending[name]
            episode.request_ids.append(request.request_id)
            episode.harmed_incarnations.add(incarnation)
            episode.last_requested_at = event.timestamp
            episode.last_applied_at = max(episode.last_applied_at, effect.timestamp)
        elif isinstance(event, SoakObservation):
            names = [cell["metadata"]["name"] for cell in event.cells or []]
            if len(names) != len(set(names)):
                raise ValueError("Recovery observation contains duplicate cell identities")
            for cell in event.cells or []:
                name = cell["metadata"]["name"]
                if name not in pending:
                    continue
                episode = pending[name]
                if _proves_recovery(
                    cell=cell, observed_at=event.timestamp, episode=episode, reconfigurations=reconfigurations
                ):
                    episode.recovered_incarnation = cell["status"]["workers_hash"]
                    episode.recovered_at = event.timestamp
                    del pending[name]
    return episodes


def _index_applied_actions(events: list[Event]) -> dict[str, SoakActionAppliedEvent]:
    requests: dict[str, SoakActionRequestedEvent] = {}
    applied: dict[str, SoakActionAppliedEvent] = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            request_id = event.request.request_id
            if request_id in requests:
                raise ValueError(f"Duplicate fault request: {request_id}")
            requests[request_id] = event
        elif isinstance(event, SoakActionAppliedEvent):
            if event.request_id not in requests or event.request_id in applied:
                raise ValueError(f"Unknown or duplicate applied action: {event.request_id}")
            if event.timestamp < requests[event.request_id].timestamp:
                raise ValueError(f"Applied action precedes its request: {event.request_id}")
            applied[event.request_id] = event
    return applied


def _proves_recovery(
    *,
    cell: dict,
    observed_at: datetime,
    episode: RecoveryEpisode,
    reconfigurations: list[CellReconfigureEvent] | None,
) -> bool:
    incarnation = cell["status"].get("workers_hash")
    if (
        observed_at < episode.last_applied_at
        or not incarnation
        or incarnation in episode.harmed_incarnations
        or cell_type_of(cell) != episode.cell_type
        or cell["status"]["phase"] != "Running"
        or not cell_is_alive(cell)
    ):
        return False
    if episode.cell_type == "rollout":
        return any(
            condition["type"] == "Serving" and condition["status"] == "True"
            for condition in cell["status"]["conditions"]
        )
    return any(
        episode.last_requested_at <= event.timestamp <= observed_at
        and event.cell_incarnations_after.get(episode.cell_id) == incarnation
        and parse_cell_id(episode.cell_id).cell_index in event.healed_cell_indices
        for event in reconfigurations or []
    )
