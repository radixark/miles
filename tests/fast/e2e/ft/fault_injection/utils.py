from unittest.mock import MagicMock

from tests.e2e.ft.conftest_ft.fault_injection import state


def cell(
    name: str,
    *,
    healthy: bool,
    cell_type: str = "actor",
    phase: str = "Running",
    workers_hash: str = "generation-0",
) -> dict:
    status = "True" if healthy else "False"
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/workers-hash": workers_hash},
        },
        "status": {"phase": phase, "conditions": [{"type": "Healthy", "status": status}]},
    }


def names(cells: list[dict]) -> set[str]:
    return {c["metadata"]["name"] for c in cells}


def mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


SERVING = state.ObservedCellState.SERVING
RUNNING_NOT_SERVING = state.ObservedCellState.RUNNING_NOT_SERVING
PENDING = state.ObservedCellState.PENDING
SUSPENDED = state.ObservedCellState.SUSPENDED


def staged(
    name: str,
    cell_state: state.ObservedCellState,
    *,
    cell_type: str = "rollout",
    workers_hash: str = "generation-0",
) -> dict:
    phase = {
        SUSPENDED: "Suspended",
        PENDING: "Pending",
        RUNNING_NOT_SERVING: "Running",
        SERVING: "Running",
    }[cell_state]
    conditions: list[dict] = (
        [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if cell_state is SERVING else "False"},
        ]
        if phase == "Running"
        else []
    )
    return {
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/workers-hash": workers_hash},
        },
        "status": {"phase": phase, "conditions": conditions},
    }


def log_of(
    cell_states: list[state.ObservedCellState], *, inject_before: dict[int, int] | None = None
) -> state.EventLog:
    log = state.EventLog()
    for index, cell_state in enumerate(cell_states):
        for _ in range((inject_before or {}).get(index, 0)):
            note_injected(log, "rollout-engine-0")
        log.observe([staged("rollout-engine-0", cell_state)])
    return log
