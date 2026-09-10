import random
from dataclasses import dataclass

from tests.utils.soak.state import SoakObservation, cell_is_alive, cell_type_of

from miles.utils.audit_utils.event_logger.models import WeightUpdateAssignmentEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.workers.cell_operations.base import FaultTarget


@dataclass(frozen=True)
class SenderBatch:
    trigger: FaultTarget
    targets: list[dict]


def choose_sender_batch(
    *,
    observation: SoakObservation,
    targets: list[dict],
    triggers: list[FaultTarget],
    min_survivors: int,
    rng: random.Random,
) -> SenderBatch | None:
    assignment = max(
        (
            event
            for event in observation.training_events
            if isinstance(event, WeightUpdateAssignmentEvent)
            and isinstance(event.source, TrainerControllerProcessIdentity)
            and event.source.trainer_id == "actor"
        ),
        key=lambda event: event.timestamp,
        default=None,
    )
    if assignment is None or len(assignment.targets_by_trainer) < 2:
        return None
    assert assignment.trainer_incarnations.keys() == assignment.targets_by_trainer.keys()
    alive = {
        cell["metadata"]["name"]: cell["status"].get("workers_hash")
        for cell in observation.cells or []
        if cell_type_of(cell) == "actor" and cell_is_alive(cell)
    }
    if any(alive.get(name) != incarnation for name, incarnation in assignment.trainer_incarnations.items()):
        return None
    available = {cell["metadata"]["name"]: cell for cell in targets}
    candidates = []
    for trigger in triggers:
        if assignment.trainer_incarnations.get(trigger.cell_id) != trigger.workers_hash:
            continue
        assigned = assignment.targets_by_trainer[trigger.cell_id]
        if len(assigned) < 2 or len(available) - len(assigned) < max(1, min_survivors):
            continue
        if any(
            name not in available or available[name]["status"].get("workers_hash") != incarnation
            for name, incarnation in assigned.items()
        ):
            continue
        candidates.append(SenderBatch(trigger=trigger, targets=[available[name] for name in assigned]))
    return rng.choice(candidates) if candidates else None
