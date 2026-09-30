from itertools import pairwise

from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check"]


class WeightMovementIssue(FrozenStrictBaseModel):
    trainer_model_id: str | None
    debug_trainer_load_state_timestamp: float
    weight_version_before: int
    weight_version_after: int
    description: str


def check(events: list[Event], *, include_latest: bool = False) -> list[WeightMovementIssue]:
    """Check: every tensor changes between adjacent published versions of one trainer load state."""
    latest = max(
        (event.timestamp for event in events if isinstance(event, (WeightUpdateResultEvent, TrainGroupStepEndEvent))),
        default=None,
    )
    versions: dict[tuple[str | None, float], dict[int, dict[str, str]]] = {}
    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        if not include_latest and (latest is None or event.timestamp >= latest):
            continue
        lineage = (event.trainer_model_id, event.debug_trainer_load_state_timestamp)
        for snapshot in event.engine_snapshots:
            versions.setdefault(lineage, {}).setdefault(event.weight_version, snapshot.tensor_checksums)

    issues: list[WeightMovementIssue] = []
    for (trainer_model_id, load_state_timestamp), by_version in versions.items():
        for before, after in pairwise(sorted(by_version)):
            previous, current = by_version[before], by_version[after]
            if previous.keys() != current.keys():
                description = "tensor set changed"
            elif unchanged := sorted(name for name in current if current[name] == previous[name]):
                description = f"unchanged tensor checksums: {unchanged}"
            else:
                continue
            issues.append(
                WeightMovementIssue(
                    trainer_model_id=trainer_model_id,
                    debug_trainer_load_state_timestamp=load_state_timestamp,
                    weight_version_before=before,
                    weight_version_after=after,
                    description=description,
                )
            )
    return issues
