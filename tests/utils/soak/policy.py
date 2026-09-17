from tests.utils.soak.config import SoakCellPolicy
from tests.utils.soak.state import SoakActionRequestedEvent, SoakEvent, cell_is_alive, cell_type_of


def eligible_cells(
    *, cells: list[dict], events: list[SoakEvent], policy: SoakCellPolicy, harms_cell: bool
) -> list[dict]:
    reserved = {
        _identity(event.request.target)
        for event in events
        if isinstance(event, SoakActionRequestedEvent)
        and event.request.harms_cell
        and isinstance(event.request.target, dict)
    }
    ready = {_identity(cell) for cell in cells if _is_ready(cell) and _identity(cell) not in reserved}
    if len(ready) != len(cells) or (policy.expected_cells is not None and len(cells) != policy.expected_cells):
        return []
    return [
        cell
        for cell in cells
        if _identity(cell) in ready
        and (not harms_cell or (_identity(cell) not in reserved and len(ready - {_identity(cell)}) >= 1))
    ]


def _identity(cell: dict) -> tuple[str, str | None]:
    return cell["metadata"]["name"], cell["status"].get("workers_hash")


def _is_ready(cell: dict) -> bool:
    if cell["status"]["phase"] != "Running" or not cell_is_alive(cell):
        return False
    return cell_type_of(cell) != "rollout" or any(
        condition["type"] == "Serving" and condition["status"] == "True" for condition in cell["status"]["conditions"]
    )
