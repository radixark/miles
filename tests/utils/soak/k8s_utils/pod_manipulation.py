import asyncio
from typing import Any, Literal

from pydantic import Field
from tests.utils.soak.k8s_utils.pod_processes import ProcessTarget

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS
from miles.utils.workers.k8s_client import core_v1_api


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
    assert pod.uid, "Pod deletion requires an observed UID"
    async with asyncio.timeout(KUBECTL_TIMEOUT_SECONDS), core_v1_api() as api:
        return await _delete_and_confirm_pod(api=api, pod=pod)


async def _delete_and_confirm_pod(*, api: Any, pod: SoakPodTarget) -> PodDeletedEvidence:
    from kubernetes_asyncio import client

    before = await api.read_namespaced_pod(name=pod.name, namespace=pod.namespace)
    assert before.metadata.uid == pod.uid, "Pod identity changed before deletion"
    assert before.metadata.deletion_timestamp is None, "Pod was already deleting before injection"
    assert before.metadata.resource_version, "Pod deletion requires a resource version"
    await api.delete_namespaced_pod(
        name=pod.name,
        namespace=pod.namespace,
        body=client.V1DeleteOptions(
            preconditions=client.V1Preconditions(uid=pod.uid, resource_version=before.metadata.resource_version)
        ),
    )
    while True:
        try:
            current = await api.read_namespaced_pod(name=pod.name, namespace=pod.namespace)
        except client.ApiException as error:
            if error.status != 404:
                raise
            break
        if current.metadata.uid != pod.uid:
            break
        await asyncio.sleep(0.2)
    return PodDeletedEvidence(namespace=pod.namespace, pod_name=pod.name, pod_uid=pod.uid)
