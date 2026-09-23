from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Discriminator, Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.actions.union import FaultAction


class FaultHookOwner(StrEnum):
    TRAINER_ACTOR = "trainer_actor"
    TRAINER_CONTROLLER = "trainer_controller"
    ORCHESTRATOR = "orchestrator"


class FaultHookName(StrEnum):
    TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER = "trainer_weight_update_before_all_gather"
    TRAINER_WEIGHT_UPDATE_BEFORE_SEND = "trainer_weight_update_before_send"
    TRAINER_STEP_BEFORE_ALLREDUCE = "trainer_step_before_allreduce"
    TRAINER_CONTROLLER_STEP_END = "trainer_controller_step_end"
    ORCHESTRATOR_STEP_END = "orchestrator_step_end"

    @property
    def owner(self) -> FaultHookOwner:
        match self:
            case (
                FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
                | FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND
                | FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE
            ):
                return FaultHookOwner.TRAINER_ACTOR
            case FaultHookName.TRAINER_CONTROLLER_STEP_END:
                return FaultHookOwner.TRAINER_CONTROLLER
            case FaultHookName.ORCHESTRATOR_STEP_END:
                return FaultHookOwner.ORCHESTRATOR


class FaultHookStatus(StrEnum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CLEARED = "cleared"
    EXPIRED = "expired"
    FIRED = "fired"
    FAILED = "failed"


class _BaseFaultHookTarget(FrozenStrictBaseModel):
    cell_id: str | None = None
    rank: int | None = Field(default=None, ge=0)

    def covers(self, *, cell_id: str | None, rank: int | None) -> bool:
        return self.cell_id in (None, cell_id) and self.rank in (None, rank)


class DeclaredFaultHookTarget(_BaseFaultHookTarget):
    kind: Literal["declared"] = "declared"


class ObservedFaultHookTarget(_BaseFaultHookTarget):
    kind: Literal["observed"] = "observed"
    cell_id: str
    rank: int = Field(ge=0)
    workers_hash: str = Field(min_length=1)
    boot_uuid: str | None = None
    pod_uid: str | None = None


FaultHookTarget = Annotated[DeclaredFaultHookTarget | ObservedFaultHookTarget, Discriminator("kind")]


class FaultHookRequest(FrozenStrictBaseModel):
    request_id: str = Field(min_length=1)
    hook_name: FaultHookName | None = None
    action: FaultAction
    target: FaultHookTarget = DeclaredFaultHookTarget()
    rollout_id: int | None = Field(default=None, ge=0)
    attempt: int | None = Field(default=None, ge=0)
    weight_version: int | None = Field(default=None, ge=0)
    lifetime_seconds: float | None = Field(default=None, gt=0, le=300, allow_inf_nan=False)
    delay_ms: float = Field(default=0.0, ge=0, le=300_000, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_target(self) -> "FaultHookRequest":
        if (
            self.hook_name is not None
            and self.hook_name.owner is not FaultHookOwner.TRAINER_ACTOR
            and self.target != DeclaredFaultHookTarget()
        ):
            raise ValueError(f"{self.hook_name} is reached outside any cell, so its request cannot name a target")
        return self

    def matches(self, context: FaultHookContext) -> bool:
        return (
            self.rollout_id in (None, context.rollout_id)
            and self.attempt in (None, context.attempt)
            and self.weight_version in (None, context.weight_version)
        )

    def conflicts_with(self, other: "FaultHookRequest") -> bool:
        return (
            other.hook_name is not None
            and self.hook_name == other.hook_name
            and self.action == other.action
            and self.rollout_id == other.rollout_id
            and self.attempt == other.attempt
            and self.weight_version == other.weight_version
        )


class FaultHookRecord(FrozenStrictBaseModel):
    request: FaultHookRequest
    status: FaultHookStatus
    set_at: float
    changed_at: float
    reached_at: float | None = None
    due_at: float | None = None
    context: FaultHookContext | None = None
