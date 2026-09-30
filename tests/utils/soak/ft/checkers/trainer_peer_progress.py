from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakEvent,
    SoakObservationEvent,
)
from tests.utils.soak.core.views import alive_targets_of_kind
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import Event, TrainGroupStepEndEvent
from miles.utils.workers.naming import parse_cell_id


def assert_trainer_peers_progress(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    steps = [event for event in training_events if isinstance(event, TrainGroupStepEndEvent)]
    observed: dict[str, str] = {}
    candidates: dict[str, dict[str, str]] = {}
    checked = 0
    for event in events:
        if isinstance(event, SoakObservationEvent) and event.targets is not None:
            observed = {
                target.identity: target.incarnation
                for target in alive_targets_of_kind(event, ACTOR_CELL_TYPE)
                if target.incarnation
            }
        elif isinstance(event, SoakActionRequestedEvent) and event.request.target.kind == ACTOR_CELL_TYPE:
            target = event.request.target.identity
            pool = parse_cell_id(target).pool_id
            candidates[event.request.request_id] = {
                name: incarnation
                for name, incarnation in observed.items()
                if name != target and parse_cell_id(name).pool_id == pool
            }
        elif isinstance(event, SoakActionAppliedEvent) and event.request_id in candidates:
            survivors = candidates[event.request_id]
            assert survivors, "No original healthy peer was observed before the trainer fault request"
            assert any(
                step.timestamp > event.timestamp
                and _peer_completed_normal_step(step, name=name, incarnation=incarnation)
                for name, incarnation in survivors.items()
                for step in steps
            ), "No original peer completed normal training after the trainer fault effect"
            checked += 1

    assert checked, "No applied trainer fault had survivor evidence"


def _peer_completed_normal_step(step: TrainGroupStepEndEvent, *, name: str, incarnation: str) -> bool:
    outcomes = step.cell_outcomes.get(parse_cell_id(name).cell_index)
    return (
        step.cell_incarnations.get(name) == incarnation
        and isinstance(outcomes, list)
        and bool(outcomes)
        and all(outcome == TrainStepOutcome.NORMAL for outcome in outcomes)
    )
