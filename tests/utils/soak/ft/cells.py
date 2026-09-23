from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.utils.ft_utils.api_server.models import CELL_TYPE_LABEL, Cell, TriState


def cell_is_ready(cell: Cell) -> bool:
    if cell.status.phase != "Running" or not cell_is_alive(cell):
        return False
    return cell_type_of(cell) != ROLLOUT_CELL_TYPE or any(
        condition.type == "Serving" and condition.status is TriState.TRUE for condition in cell.status.conditions
    )


def cell_is_alive(cell: Cell) -> bool:
    return any(
        condition.type == "Healthy" and condition.status is TriState.TRUE for condition in cell.status.conditions
    )


def cell_type_of(cell: Cell) -> str:
    return cell.metadata.labels[CELL_TYPE_LABEL]
