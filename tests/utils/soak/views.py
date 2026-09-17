# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from dataclasses import dataclass, replace
from datetime import datetime

from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakDeploymentTarget,
    SoakEvent,
    target_type_of,
)


@dataclass(frozen=True)
class SoakActionRecord:
    requested: SoakActionRequestedEvent
    applied: SoakActionAppliedEvent | None = None
    result: SoakActionResultEvent | None = None


def project_actions(events: list[SoakEvent]) -> dict[str, SoakActionRecord]:
    actions: dict[str, SoakActionRecord] = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            request_id = event.request.request_id
            if request_id in actions:
                raise ValueError(f"Duplicate soak request: {request_id}")
            actions[request_id] = SoakActionRecord(requested=event)
        elif isinstance(event, SoakActionAppliedEvent):
            if event.request_id not in actions:
                raise ValueError(f"Soak application without request: {event.request_id}")
            action = actions[event.request_id]
            if action.applied is not None:
                raise ValueError(f"Duplicate soak application: {event.request_id}")
            actions[event.request_id] = replace(action, applied=event)
        elif isinstance(event, SoakActionResultEvent):
            if event.request_id not in actions:
                raise ValueError(f"Soak result without request: {event.request_id}")
            action = actions[event.request_id]
            if action.result is not None:
                raise ValueError(f"Duplicate soak result: {event.request_id}")
            actions[event.request_id] = replace(action, result=event)
    return actions


def compute_num_injections(events: list[SoakEvent], *, cell_type: str | None = None, harmed_only: bool = True) -> int:
    return sum(
        1
        for request, outcome in _compute_action_outcomes(events)
        if isinstance(outcome, SoakActionAppliedEvent)
        and not isinstance(request.target, SoakDeploymentTarget)
        and (cell_type is None or target_type_of(request.target) == cell_type)
        and (request.harms_cell or not harmed_only)
    )


def compute_injection_times(events: list[SoakEvent], *, cell_type: str | None = None) -> list[datetime]:
    return [
        outcome.timestamp
        for request, outcome in _compute_action_outcomes(events)
        if isinstance(outcome, SoakActionAppliedEvent)
        and not isinstance(request.target, SoakDeploymentTarget)
        and (cell_type is None or target_type_of(request.target) == cell_type)
    ]


def compute_successful_form_names(events: list[SoakEvent], *, cell_type: str) -> set[str]:
    return {
        request.form_name
        for request, outcome in _compute_action_outcomes(events)
        if isinstance(outcome, SoakActionAppliedEvent) and target_type_of(request.target) == cell_type
    }


def compute_forms_drawn_without_success(events: list[SoakEvent]) -> list[tuple[str, str]]:
    drawn: set[tuple[str, str]] = set()
    worked: set[tuple[str, str]] = set()
    for request, outcome in _compute_action_outcomes(events):
        if isinstance(request.target, SoakDeploymentTarget):
            continue
        key = (target_type_of(request.target), request.form_name)
        drawn.add(key)
        if isinstance(outcome, SoakActionAppliedEvent):
            worked.add(key)
    return sorted(drawn - worked)


def _compute_action_outcomes(
    events: list[SoakEvent],
) -> list[tuple[SoakActionRequest, SoakActionAppliedEvent | SoakActionResultEvent]]:
    actions = project_actions(events)
    applied: set[str] = set()
    outcomes: list[tuple[SoakActionRequest, SoakActionAppliedEvent | SoakActionResultEvent]] = []
    for event in events:
        if isinstance(event, SoakActionResultEvent):
            if not event.returned and event.request_id not in applied:
                outcomes.append((actions[event.request_id].requested.request, event))
        elif isinstance(event, SoakActionAppliedEvent):
            applied.add(event.request_id)
            outcomes.append((actions[event.request_id].requested.request, event))
    return outcomes
