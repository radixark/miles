from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import Discriminator, Field, model_validator

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.checksum_policy import ChecksumMovementSkipReason
from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksumSnapshot
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class EnvReportEditablePackageInfo(FrozenStrictBaseModel):
    name: str
    version: str
    location: str


class EnvReportGitRepoInfo(FrozenStrictBaseModel):
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str
    uncommitted_hash: str | None
    untracked_paths: list[str]
    untracked_paths_truncated: bool
    untracked_unhashed_paths: list[str]


class EnvReportArgsDump(FrozenStrictBaseModel):
    values: dict[str, Any]
    skipped_names: list[str]


class EnvReportProcessFacts(FrozenStrictBaseModel):
    hostname: str
    argv: list[str]
    args: EnvReportArgsDump
    env_vars: dict[str, str]
    launcher_env_report: dict[str, Any] | None


class EnvReport(FrozenStrictBaseModel):
    process: EnvReportProcessFacts
    key_versions: dict[str, str]
    editable_packages: list[EnvReportEditablePackageInfo]
    git_repos: list[EnvReportGitRepoInfo]
    full_pip_list: list[dict[str, str]]
    packages_probed: bool


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


class TrainEngineLocalWeightChecksumState(FrozenStrictBaseModel):
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    # May be skipped in non-debug mode if too expensive
    optimizer_hashes: list[OptimizerStateInfo]


class TrainEngineLocalWeightChecksumEvent(_ActorTrainEventBase):
    type: Literal["train_engine_local_weight_checksum"] = "train_engine_local_weight_checksum"
    state: TrainEngineLocalWeightChecksumState


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
    cell_incarnations: dict[str, str] = Field(default_factory=dict)


class CellReconfigureEvent(EventBase):
    type: Literal["cell_reconfigure"] = "cell_reconfigure"
    rollout_id: int
    quorum_id: int
    src_cell_index: int | None
    # healing happened iff non-empty
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]
    cell_incarnations_after: dict[str, str] = Field(default_factory=dict)


class InferenceEngineWeightChecksumEvent(EventBase):
    type: Literal["inference_engine_weight_checksum"] = "inference_engine_weight_checksum"
    # The out-of-loop startup sync stamps start_rollout_id - 1, so -1 is a fresh run's initial sync.
    rollout_id: int
    # The policy whose weights were pushed, or None for a run that trains one unnamed policy.
    trainer_model_id: str | None = None
    # One {tensor -> hash} dict per rollout engine; a TP>1 engine's ranks merge with a rank{r}/ prefix.
    engine_checksums: list[dict[str, str]]
    weight_version: int | None = None
    engine_snapshots: list[InferenceEngineChecksumSnapshot] = Field(default_factory=list)
    movement_skip_reasons: list[ChecksumMovementSkipReason] | None = None

    @model_validator(mode="after")
    def _validate_snapshot_identity(self) -> Self:
        if self.weight_version is None and not self.engine_snapshots:
            return self
        if self.weight_version is None or not self.engine_snapshots:
            raise ValueError("Checksum version and snapshots must be recorded together")
        if self.engine_checksums != [snapshot.tensors for snapshot in self.engine_snapshots]:
            raise ValueError("Checksum tensors disagree with their identified snapshots")
        cell_ids = [snapshot.cell_id for snapshot in self.engine_snapshots]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Checksum event contains duplicate cell identities")
        if len({snapshot.model_name for snapshot in self.engine_snapshots}) != 1:
            raise ValueError("Checksum event mixes inference models")
        return self


class TrainAdvantageComputationEvent(_ActorTrainEventBase):
    type: Literal["train_advantage_computation"] = "train_advantage_computation"
    advantages: list[list[float]]
    witness_ids: list[list[int]]


class EnvReportEvent(EventBase):
    type: Literal["env_report"] = "env_report"
    report: EnvReport


class EngineEnvReportEvent(EventBase):
    type: Literal["engine_env_report"] = "engine_env_report"
    cell_id: str
    workers_hash: str | None = None
    server_url: str
    server_info: dict[str, Any]


class MetricEvent(EventBase):
    type: Literal["metric"] = "metric"
    rollout_id: int | None = None
    attempt: int | None = None
    evaluation_started_at: datetime | None = None
    metrics: dict[str, Any]


class FaultHookEvent(EventBase):
    type: Literal["fault_hook"] = "fault_hook"
    request_id: str
    instance_id: str
    hook: str
    mode: str
    action: Literal["inject", "observe"] = "inject"
    status: Literal["armed", "scheduled", "cancelled", "expired", "fired", "failed"]
    monotonic_time: float
    reached_at: float | None = None
    due_at: float | None = None
    rollout_id: int | None = None
    attempt: int | None = None
    weight_version: int | None = None
    update_id: str | None = None
    target_incarnations: dict[str, str] = Field(default_factory=dict)


class WeightUpdateAssignmentEvent(EventBase):
    type: Literal["weight_update_assignment"] = "weight_update_assignment"
    update_id: str
    candidate_version: int
    trainer_incarnations: dict[str, str]
    targets_by_trainer: dict[str, dict[str, str]]


class WeightUpdateResultEvent(EventBase):
    type: Literal["weight_update_result"] = "weight_update_result"
    update_id: str
    rollout_id: int | None
    candidate_version: int
    published_version: int | None
    target_incarnations: dict[str, str]
    updated_cell_ids: list[str]
    failed_cell_ids: list[str]


Event = Annotated[
    TrainEngineLocalWeightChecksumEvent
    | WitnessSnapshotParamEvent
    | WitnessAllocateIdEvent
    | TrainGroupStepEndEvent
    | CellReconfigureEvent
    | InferenceEngineWeightChecksumEvent
    | TrainAdvantageComputationEvent
    | EnvReportEvent
    | EngineEnvReportEvent
    | MetricEvent
    | FaultHookEvent
    | WeightUpdateAssignmentEvent
    | WeightUpdateResultEvent,
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
