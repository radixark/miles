from tests.utils.soak.config import SoakCellPolicy
from tests.utils.soak.state import (
    Event,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakDeploymentTarget,
    cell_is_alive,
    cell_type_of,
)


def pending_actions(events: list[Event]) -> list[SoakActionRequest]:
    pending: dict[str, SoakActionRequest] = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            pending[event.request.request_id] = event.request
        elif isinstance(event, SoakActionResultEvent):
            pending.pop(event.request_id, None)
        elif isinstance(event, SoakActionAppliedEvent):
            request = pending.get(event.request_id)
            if request is not None and isinstance(request.target, SoakDeploymentTarget):
                del pending[event.request_id]
    return list(pending.values())


def eligible_cells(*, cells: list[dict], events: list[Event], policy: SoakCellPolicy, harms_cell: bool) -> list[dict]:
    reserved = {
        _identity(event.request.target)
        for event in events
        if isinstance(event, SoakActionRequestedEvent)
        and event.request.harms_cell
        and isinstance(event.request.target, dict)
    }
    ready = {_identity(cell) for cell in cells if _is_ready(cell) and _identity(cell) not in reserved}
    if not policy.allow_during_recovery and (
        len(cells) != policy.expected_cells or len(ready) != policy.expected_cells
    ):
        return []
    return [
        cell
        for cell in cells
        if (not policy.require_ready_target or _identity(cell) in ready)
        and (
            not harms_cell
            or (_identity(cell) not in reserved and len(ready - {_identity(cell)}) >= policy.min_survivors)
        )
    ]


def _identity(cell: dict) -> tuple[str, str | None]:
    return cell["metadata"]["name"], cell["status"].get("workers_hash")


def _is_ready(cell: dict) -> bool:
    if cell["status"]["phase"] != "Running" or not cell_is_alive(cell):
        return False
    return cell_type_of(cell) != "rollout" or any(
        condition["type"] == "Serving" and condition["status"] == "True" for condition in cell["status"]["conditions"]
    )
