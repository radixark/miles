from enum import StrEnum
from typing import Literal

from pydantic import Field
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget

from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import ObservedFaultHookTarget

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
