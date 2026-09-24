from enum import StrEnum
from typing import Literal

from pydantic import Field
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName, ObservedFaultHookTarget

ACTOR_CELL_TYPE: str = ACTOR_ROLE
ROLLOUT_CELL_TYPE: str = "rollout"
POOL_TARGET_KIND: str = "pool"

Moment = Literal["generating", "training"]


class FaultTrigger(StrEnum):
    TIMER = "timer"
    HOOK = "hook"


class CellTarget(FrozenStrictBaseModel):
    kind: Literal["actor", "rollout"]
    identity: str
    incarnation: str
    alive: bool
    ready: bool
    pods: list[SoakPodTarget] = Field(default_factory=list)
    fault_target: ObservedFaultHookTarget | None = None


class PoolTarget(FrozenStrictBaseModel):
    kind: Literal["pool"] = "pool"
    identity: str
    incarnation: str = ""
    alive: bool
    ready: bool
    cell_type: str
    replicas: int
    cell_ids: list[str]


class InjectFaultDetails(FrozenStrictBaseModel):
    form: Literal["inject_fault"] = "inject_fault"
    fault_target: ObservedFaultHookTarget
    hook_target: ObservedFaultHookTarget
    hook_name: FaultHookName | None = None
    delay_ms: float = Field(default=0, ge=0, le=300_000, allow_inf_nan=False)


class PodDetails(FrozenStrictBaseModel):
    form: Literal["pod"] = "pod"
    pod: SoakPodTarget


class ResizeDetails(FrozenStrictBaseModel):
    form: Literal["resize"] = "resize"
    replicas: int
    at_rollout: int | None
    moment: Moment | None


class ObservedCellFaultKind(StrEnum):
    MISSING = "missing"
    REPLACED = "replaced"
    UNHEALTHY = "unhealthy"


class ObservedCellFault(FrozenStrictBaseModel):
    kind: Literal["cell_fault"] = "cell_fault"
    request_id: str
    target: ObservedFaultHookTarget
    action: FaultAction
    observed: ObservedCellFaultKind
    observed_workers_hash: str | None = None


class PoolResizedEvidence(FrozenStrictBaseModel):
    kind: Literal["pool_resized"] = "pool_resized"
    replicas_before: int
    replicas_after: int
