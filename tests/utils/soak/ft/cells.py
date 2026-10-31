from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.utils.ft_utils.api_server.models import CELL_TYPE_LABEL, Cell, TriState


def cell_is_ready(cell: Cell) -> bool:
    if cell.status.phase != "Running" or not _has_condition(cell, "Healthy", TriState.TRUE):
        return False
    return cell_type_of(cell) != ROLLOUT_CELL_TYPE or _has_condition(cell, "Serving", TriState.TRUE)


def cell_is_alive(cell: Cell) -> bool:
    return (
        cell.status.phase == "Running"
        and any(condition.type == "Healthy" for condition in cell.status.conditions)
        and not _has_condition(cell, "Healthy", TriState.FALSE)
    )


def cell_type_of(cell: Cell) -> str:
    return cell.metadata.labels[CELL_TYPE_LABEL]


def _has_condition(cell: Cell, condition_type: str, status: TriState) -> bool:
    return any(condition.type == condition_type and condition.status is status for condition in cell.status.conditions)
