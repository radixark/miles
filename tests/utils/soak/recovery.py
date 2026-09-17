from dataclasses import dataclass, field
from datetime import datetime

from tests.utils.soak.state import SoakActionRequestedEvent, SoakEvent, SoakObservation, cell_is_alive, cell_type_of
from tests.utils.soak.views import project_actions

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
    events: list[SoakEvent],
    *,
    reconfigurations: list[CellReconfigureEvent] | None = None,
) -> list[RecoveryEpisode]:
    actions = project_actions(events)
    for request_id, action in actions.items():
        if action.applied is not None and action.applied.timestamp < action.requested.timestamp:
            raise ValueError(f"Applied action precedes its request: {request_id}")

    episodes: list[RecoveryEpisode] = []
    pending: dict[str, RecoveryEpisode] = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            request = event.request
            effect = actions[request.request_id].applied
            if effect is None or not request.harms_cell or not isinstance(request.target, dict):
                continue
            name = request.target["metadata"]["name"]
            incarnation = request.target["status"]["workers_hash"]
            if not incarnation:
                raise ValueError("Recovery evidence requires a nonempty target incarnation")
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
