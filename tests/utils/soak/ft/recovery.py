from datetime import datetime

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import SoakActionRecord, sut_events
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, CellTarget

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.workers.naming import parse_cell_id


def compute_recovered_at(*, action: SoakActionRecord, events: list[SoakEvent]) -> datetime | None:
    request = action.requested.request
    if action.applied is None:
        return None
    assert request.target.incarnation, "Recovery evidence requires a nonempty target incarnation"

    reconfigurations = [event for event in sut_events(events) if isinstance(event, CellReconfigureEvent)]
    for observation in events:
        if not isinstance(observation, SoakObservationEvent) or observation.timestamp < action.applied.timestamp:
            continue
        for target in observation.targets or []:
            if (
                target.identity == request.target.identity
                and target.kind == request.target.kind
                and target.incarnation
                and target.incarnation != request.target.incarnation
                and target.ready
                and _healed_into(
                    target,
                    observed_at=observation.timestamp,
                    requested_at=action.requested.timestamp,
                    reconfigurations=reconfigurations,
                )
            ):
                return observation.timestamp
    return None


def _healed_into(
    target: CellTarget,
    *,
    observed_at: datetime,
    requested_at: datetime,
    reconfigurations: list[CellReconfigureEvent],
) -> bool:
    if target.kind != ACTOR_CELL_TYPE:
        return True
    return any(
        requested_at < event.timestamp <= observed_at
        and parse_cell_id(target.identity).cell_index in event.healed_cell_indices
        and event.cell_incarnations_after.get(target.identity) == target.incarnation
        for event in reconfigurations
    )
