from typing import Literal

from pydantic import Field
from tests.utils.soak.k8s_utils.pod_processes import ProcessTarget

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PodDeletedEvidence(FrozenStrictBaseModel):
    kind: Literal["pod_deleted"] = "pod_deleted"
    namespace: str
    pod_name: str
    pod_uid: str


class SoakPodTarget(FrozenStrictBaseModel):
    namespace: str
    release: str
    name: str
    uid: str
    process_targets: dict[str, ProcessTarget] = Field(default_factory=dict)


async def delete_observed_pod(pod: SoakPodTarget) -> PodDeletedEvidence:
    raise NotImplementedError
