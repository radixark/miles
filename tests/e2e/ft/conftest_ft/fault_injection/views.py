# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import dataclasses
from datetime import datetime
from typing import Literal

from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent, ObservationsEvent, ObservedCellState

STALE_STATUS_GRACE_SECONDS: float = 120.0


def compute_num_injections(events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True) -> int:
    return len(compute_injected_cell_names(events, cell_type=cell_type, harmed_only=harmed_only))


def compute_injected_cell_names(
    events: list[Event], *, cell_type: str | None = None, harmed_only: bool = True
) -> list[str]:
    return [
        name
        for name, cell_events in _compute_matching_cell_events(
            events, cell_type=cell_type, harmed_only=harmed_only
        ).items()
        for one in cell_events
        if one.kind == "injected"
    ]


def compute_num_successful_injections_of_form(events: list[Event], *, form_name: str) -> int:
    return len(
        [
            event
            for event in events
            if isinstance(event, InjectionEvent) and event.succeeded and event.form_name == form_name
        ]
    )


def compute_cells_not_serving_after_injection(
    events: list[Event], *, cell_type: str, grace_seconds: float | None = None
) -> dict[str, list[str]]:
    if grace_seconds is None:
        grace_seconds = STALE_STATUS_GRACE_SECONDS

    cell_type_of_name = _compute_cell_type_of_name(events)
    last_injection_time_of_name: dict[str, datetime] = {
        event.cell_name: event.timestamp
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and event.harmed
        and cell_type_of_name.get(event.cell_name) == cell_type
    }

    served: set[str] = set()
    for event in events:
        if not isinstance(event, ObservationsEvent):
            continue
        for name, injected_at in last_injection_time_of_name.items():
            info = event.cell_infos.get(name)
            if (
                name not in served
                and info is not None
                and info.state is ObservedCellState.SERVING
                and (event.timestamp - injected_at).total_seconds() >= grace_seconds
            ):
                served.add(name)

    observed_states = compute_states_of_cell_name(events)
    return {
        name: [one.value for one in observed_states.get(name, [])]
        for name in sorted(set(last_injection_time_of_name) - served)
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


def compute_injection_times(events: list[Event], *, cell_type: str | None = None) -> list[datetime]:
    cell_type_of_name = _compute_cell_type_of_name(events)
    return [
        event.timestamp
        for event in events
        if isinstance(event, InjectionEvent)
        and event.succeeded
        and (cell_type is None or cell_type_of_name.get(event.cell_name) == cell_type)
    ]


def compute_states_of_cell_name(events: list[Event]) -> dict[str, list[ObservedCellState]]:
    return {
        name: states
        for name, cell_events in _compute_cell_events(events).items()
        if (states := _compute_distinct_states(cell_events))
    }


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


def _compute_cell_events(events: list[Event], *, harmed_only: bool = True) -> dict[str, list[_CellEvent]]:
    cell_events_of_name: dict[str, list[_CellEvent]] = {}
    for event in events:
        if isinstance(event, InjectionEvent):
            if event.succeeded and (event.harmed or not harmed_only):
                cell_events_of_name.setdefault(event.cell_name, []).append(_CellEvent(kind="injected"))
            continue
        for name, info in event.cell_infos.items():
            cell_events_of_name.setdefault(name, []).append(_CellEvent(kind="observed", state=info.state))
    return cell_events_of_name


def _compute_matching_cell_events(
    events: list[Event], *, cell_type: str | None, harmed_only: bool
) -> dict[str, list[_CellEvent]]:
    cell_events_of_name = _compute_cell_events(events, harmed_only=harmed_only)
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


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states
