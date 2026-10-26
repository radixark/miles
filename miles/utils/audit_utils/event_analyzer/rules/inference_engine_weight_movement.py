from datetime import datetime
from itertools import pairwise
from typing import Any, Literal

from miles.utils.audit_utils.event_analyzer.rules.utils import trainer_args
from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

__all__ = ["check"]

_APPLICABILITY_ARG_NAMES: tuple[str, ...] = ("lora_rank", "lora_adapter_path", "update_weights_interval")


class WeightMovementIssue(FrozenStrictBaseModel):
    trainer_model_id: str | None
    debug_trainer_load_state_timestamp: float
    weight_version_before: int
    weight_version_after: int
    kind: Literal["tensor_set_changed", "unchanged_tensors", "unchanged_weights"]
    description: str

    @property
    def is_warning_only(self) -> bool:
        return self.kind == "unchanged_tensors"


def check(events: list[Event], *, include_latest: bool = False) -> list[WeightMovementIssue]:
    """Check: every tensor changes between adjacent published versions of one trainer load state."""
    latest = max(
        (event.timestamp for event in events if isinstance(event, (WeightUpdateResultEvent, TrainGroupStepEndEvent))),
        default=None,
    )
    versions: dict[tuple[str | None, float], dict[int, dict[str, str]]] = {}
    first_observed_at: dict[tuple[str | None, float], datetime] = {}
    for event in sorted(events, key=lambda event: event.timestamp):
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        if not include_latest and (latest is None or event.timestamp >= latest):
            continue
        lineage = (event.trainer_model_id, event.debug_trainer_load_state_timestamp)
        first_observed_at.setdefault(lineage, event.timestamp)
        for snapshot in event.engine_snapshots:
            versions.setdefault(lineage, {}).setdefault(event.weight_version, snapshot.tensor_checksums)

    issues: list[WeightMovementIssue] = []
    for (trainer_model_id, load_state_timestamp), by_version in versions.items():
        args = trainer_args(
            events,
            names=_APPLICABILITY_ARG_NAMES,
            at=first_observed_at[trainer_model_id, load_state_timestamp],
            model_id=trainer_model_id,
        )
        if args is None or not _applies(args):
            continue
        for before, after in pairwise(sorted(by_version)):
            previous, current = by_version[before], by_version[after]
            if previous.keys() != current.keys():
                kind, description = "tensor_set_changed", "tensor set changed"
            elif unchanged := sorted(name for name in current if current[name] == previous[name]):
                if len(unchanged) == len(current):
                    kind, description = "unchanged_weights", "all tensor checksums are unchanged"
                else:
                    kind, description = "unchanged_tensors", f"unchanged tensor checksums: {unchanged}"
            else:
                continue
            issues.append(
                WeightMovementIssue(
                    trainer_model_id=trainer_model_id,
                    debug_trainer_load_state_timestamp=load_state_timestamp,
                    weight_version_before=before,
                    weight_version_after=after,
                    kind=kind,
                    description=description,
                )
            )
    return issues


def _applies(args: dict[str, Any]) -> bool:
    return args["lora_rank"] == 0 and args["lora_adapter_path"] is None and args["update_weights_interval"] == 1
