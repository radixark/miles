# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import dataclasses
import enum
from typing import Literal

from tests.e2e.ft.conftest_ft.fault_injection.state import (
    Event,
    InjectionEvent,
    ObservationsEvent,
    ObservedCellState,
    cell_is_alive,
)

_RELAUNCH_STATES: tuple[ObservedCellState, ...] = (ObservedCellState.SUSPENDED, ObservedCellState.PENDING)


def compute_genuinely_alive(events: list[Event], cells: list[dict]) -> list[dict]:
    awaiting = compute_cells_awaiting_recovery(events)
    return [c for c in cells if cell_is_alive(c) and c["metadata"]["name"] not in awaiting]


def compute_cells_awaiting_recovery(events: list[Event]) -> set[str]:
    state_of_cell_name: dict[str, _CellState] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.succeeded:
                state_of_cell_name[event.cell_name] = _CellState.INJECTED
            continue
        for name, state in list(state_of_cell_name.items()):
            info = event.cell_infos.get(name)
            if info is None or not info.alive:
                state_of_cell_name[name] = _CellState.RECOVERING
            elif state is _CellState.RECOVERING:
                del state_of_cell_name[name]
    return set(state_of_cell_name)


def compute_num_injections(events: list[Event], *, cell_type: str | None = None) -> int:
    return sum(
        sum(1 for one in cell_events if one.kind == "injected")
        for cell_events in _compute_matching_cell_events(events, cell_type=cell_type).values()
    )


def compute_num_completed_recoveries(events: list[Event], *, cell_type: str | None = None) -> int:
    return sum(
        _compute_recovery_tally(cell_events).num_completed
        for cell_events in _compute_matching_cell_events(events, cell_type=cell_type).values()
    )


def compute_cells_with_unfinished_recovery(events: list[Event], *, cell_type: str | None = None) -> dict[str, int]:
    return {
        name: tally.num_unfinished
        for name, cell_events in _compute_matching_cell_events(events, cell_type=cell_type).items()
        if (tally := _compute_recovery_tally(cell_events)).num_unfinished
    }


def compute_successful_form_names(events: list[Event], *, cell_type: str) -> set[str]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        event.form_name
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and cell_type_of_name.get(event.cell_name) == cell_type
    }


def compute_forms_drawn_without_success(events: list[Event]) -> list[tuple[str, str]]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    drawn: set[tuple[str, str]] = set()
    worked: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, InjectionEvent):
            continue
        key = (cell_type_of_name.get(event.cell_name, ""), event.form_name)
        drawn.add(key)
        if event.succeeded:
            worked.add(key)
    return sorted(drawn - worked)


def compute_states_of_cell_name(events: list[Event]) -> dict[str, list[ObservedCellState]]:
    return {
        name: states
        for name, cell_events in _compute_cell_events(events).items()
        if (states := _compute_distinct_states(cell_events))
    }


class _CellState(enum.Enum):
    INJECTED = enum.auto()  # we crashed it; the api server may still report it Healthy
    RECOVERING = enum.auto()  # observed unhealthy; awaiting its return to Healthy


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


def _compute_cell_events(events: list[Event]) -> dict[str, list[_CellEvent]]:
    cell_events_of_name: dict[str, list[_CellEvent]] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.succeeded:
                cell_events_of_name.setdefault(event.cell_name, []).append(_CellEvent(kind="injected"))
            continue
        for name, info in event.cell_infos.items():
            cell_events_of_name.setdefault(name, []).append(_CellEvent(kind="observed", state=info.state))
    return cell_events_of_name


def _compute_matching_cell_events(events: list[Event], *, cell_type: str | None) -> dict[str, list[_CellEvent]]:
    cell_events_of_name = _compute_cell_events(events)
    if cell_type is None:
        return cell_events_of_name
    cell_type_of_name = _compute_cell_type_of_name(events)
    return {
        name: cell_events
        for name, cell_events in cell_events_of_name.items()
        if cell_type_of_name.get(name) == cell_type
    }


def _compute_cell_type_of_name(events: list[Event]) -> dict[str, str]:
    cell_type_of_name: dict[str, str] = {}
    for event in events:
        if isinstance(event, ObservationsEvent):
            cell_type_of_name.update({name: info.cell_type for name, info in event.cell_infos.items()})
    return cell_type_of_name


@dataclasses.dataclass(frozen=True)
class _RecoveryTally:
    num_completed: int
    num_unfinished: int


class _RecoveryStage(enum.Enum):
    AWAITING_RELAUNCH = enum.auto()
    AWAITING_SERVING = enum.auto()


def _compute_recovery_tally(events: list[_CellEvent]) -> _RecoveryTally:
    pending: list[_RecoveryStage] = []
    num_completed = 0
    for event in events:
        if event.kind == "injected":
            pending.append(_RecoveryStage.AWAITING_RELAUNCH)
            continue
        if not pending:
            continue
        if pending[0] is _RecoveryStage.AWAITING_RELAUNCH and event.state in _RELAUNCH_STATES:
            pending[0] = _RecoveryStage.AWAITING_SERVING
        elif pending[0] is _RecoveryStage.AWAITING_SERVING and event.state is ObservedCellState.SERVING:
            pending.pop(0)
            num_completed += 1
    return _RecoveryTally(num_completed=num_completed, num_unfinished=len(pending))


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states
