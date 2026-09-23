from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Discriminator, Field
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence, SoakPodTarget
from tests.utils.soak.k8s_utils.pod_processes import ProcessSignalReceipt

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName, ObservedFaultHookTarget

ACTOR_CELL_TYPE: str = ACTOR_ROLE
ROLLOUT_CELL_TYPE: str = "rollout"


class CellTarget(FrozenStrictBaseModel):
    kind: Literal["actor", "rollout"]
    identity: str
    incarnation: str
    alive: bool
    ready: bool
    pods: list[SoakPodTarget] = Field(default_factory=list)
    fault_target: ObservedFaultHookTarget | None = None


class InjectFaultDetails(FrozenStrictBaseModel):
    form: Literal["inject_fault"] = "inject_fault"
    fault_target: ObservedFaultHookTarget


class PodDetails(FrozenStrictBaseModel):
    form: Literal["pod"] = "pod"
    pod: SoakPodTarget


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


WallClockFaultDetails = InjectFaultDetails | PodDetails

WallClockFaultEvidence = ObservedCellFault | PodDeletedEvidence | ProcessSignalReceipt


class HookFaultDetails(FrozenStrictBaseModel):
    form: Literal["hook"] = "hook"
    fault_target: ObservedFaultHookTarget
    hook_name: FaultHookName
    action: FaultAction
    delay_ms: float = Field(ge=0, le=300_000, allow_inf_nan=False)


class RemoteHookFaultDetails(FrozenStrictBaseModel):
    form: Literal["remote_hook"] = "remote_hook"
    trigger: ObservedFaultHookTarget
    hook_name: FaultHookName
    delay_ms: float = Field(ge=0, le=300_000, allow_inf_nan=False)
    victim_form: str
    victim: Annotated[WallClockFaultDetails, Discriminator("form")]


class HookFaultEvidence(FrozenStrictBaseModel):
    kind: Literal["hook_fault"] = "hook_fault"
    hook_request_id: str
    effect: ObservedCellFault


class RemoteHookFaultEvidence(FrozenStrictBaseModel):
    kind: Literal["remote_hook_fault"] = "remote_hook_fault"
    hit: FaultHookEvent
    victim_evidence: Annotated[WallClockFaultEvidence, Discriminator("kind")]


CellFaultDetails = WallClockFaultDetails | HookFaultDetails | RemoteHookFaultDetails

CellFaultEvidence = WallClockFaultEvidence | HookFaultEvidence | RemoteHookFaultEvidence
