from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class EventBase(FrozenStrictBaseModel):
    timestamp: datetime
    source: ProcessIdentity


class _ActorTrainEventBase(EventBase):
    rollout_id: int
    attempt: int = 0


class OptimizerStateInfo(FrozenStrictBaseModel):
    """Snapshot of one sub-optimizer's state with tensors replaced by hashes."""

    param_names: dict[int, str]
    state_dict: dict[str, Any]


class LocalWeightChecksumState(FrozenStrictBaseModel):
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    # May be skipped in non-debug mode if too expensive
    optimizer_hashes: list[OptimizerStateInfo]


class LocalWeightChecksumEvent(_ActorTrainEventBase):
    type: Literal["local_weight_checksum"] = "local_weight_checksum"
    state: LocalWeightChecksumState


class WitnessSnapshotParamEvent(_ActorTrainEventBase):
    type: Literal["witness_snapshot_param"] = "witness_snapshot_param"
    instance_id: str
    # TODO: may shrink a contiguous range of numbers into a pair, if this is too large/slow
    nonzero_witness_ids: list[int]
    stale_ids: list[int]


class WitnessAllocateIdEvent(EventBase):
    type: Literal["witness_allocate_id"] = "witness_allocate_id"
    rollout_id: int
    attempt: int
    witness_id_to_sample_index: dict[int, int]
    # Allocator counter after this allocation; a resumed run recovers the allocator from it.
    counter_after: int


class TrainGroupStepEndEvent(EventBase):
    type: Literal["train_group_step_end"] = "train_group_step_end"
    rollout_id: int
    cell_outcomes: dict[int, Literal["error"] | list[TrainStepOutcome]]


class CellReconfigureEvent(EventBase):
    type: Literal["cell_reconfigure"] = "cell_reconfigure"
    rollout_id: int
    quorum_id: int
    src_cell_index: int | None
    # healing happened iff non-empty
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]


class TrainAdvantageComputationEvent(_ActorTrainEventBase):
    type: Literal["train_advantage_computation"] = "train_advantage_computation"
    advantages: list[list[float]]
    witness_ids: list[list[int]]


class MetricEvent(EventBase):
    type: Literal["metric"] = "metric"
    rollout_id: int | None = None
    attempt: int | None = None
    metrics: dict[str, Any]


Event = Annotated[
    LocalWeightChecksumEvent
    | WitnessSnapshotParamEvent
    | WitnessAllocateIdEvent
    | TrainGroupStepEndEvent
    | CellReconfigureEvent
    | TrainAdvantageComputationEvent
    | MetricEvent,
    Discriminator("type"),
]


def _to_snake_case(name: str) -> str:
    import re

    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def _check_event_naming() -> None:
    import typing

    event_types = typing.get_args(typing.get_args(Event)[0])
    for cls in event_types:
        type_value = cls.model_fields["type"].default
        expected_snake = type_value + "_event"
        actual_snake = _to_snake_case(cls.__name__)
        assert actual_snake == expected_snake, (
            f"Event class {cls.__name__} (snake: {actual_snake}) does not match "
            f"type '{type_value}' (expected snake: {expected_snake})"
        )


_check_event_naming()
