# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import asyncio
import logging
from typing import Any

from tests.utils.soak.state import SoakPodTarget

from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


async def delete_observed_pod(pod: SoakPodTarget) -> dict:
    from kubernetes_asyncio import client, config

    assert pod.uid, "Pod deletion requires an observed UID"
    async with asyncio.timeout(KUBECTL_TIMEOUT_SECONDS):
        try:
            config.load_incluster_config()
        except config.ConfigException:
            logger.debug("Loading kubeconfig outside a Kubernetes pod", exc_info=True)
            await config.load_kube_config()

        async with client.ApiClient() as api_client:
            return await _delete_and_confirm_pod(api=client.CoreV1Api(api_client), pod=pod)


async def _delete_and_confirm_pod(*, api: Any, pod: SoakPodTarget) -> dict:
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
    return {"kind": "pod_deleted", "namespace": pod.namespace, "pod_name": pod.name, "pod_uid": pod.uid}
