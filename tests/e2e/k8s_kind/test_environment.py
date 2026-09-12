# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from kubernetes_asyncio import client as kubernetes_client
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.utils import BUSYBOX_IMAGE, CELL_LABEL, pod_body, unique_name
from tests.e2e.k8s_kind.kind_cluster import KindCluster

register_cpu_ci(est_time=480, suite="stage-b-cpu", labels=[])

_STARTUP_TIMEOUT = 180.0
_TERMINATION_TIMEOUT = 120.0


class TestKindClusterEnvironment:
    async def test_a_pod_actually_reaches_running(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """The cluster has a working kubelet: a created pod is scheduled, pulled, and started."""
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(name="pod-env", cell="cell-env", image=BUSYBOX_IMAGE, command=["sleep", "3600"]),
        )

        async def pod_is_running() -> bool:
            pod = await cluster_core_v1.read_namespaced_pod(namespace=cluster_namespace, name="pod-env")
            return pod.status.phase == "Running"

        await _wait_until_async(
            pod_is_running,
            description="the pod to reach Running under a real kubelet",
            timeout=_STARTUP_TIMEOUT,
        )

    async def test_a_namespace_deletion_completes(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, kind_cluster: KindCluster
    ) -> None:
        """The controller-manager runs the namespace finalizer, unlike the bare apiserver layer."""
        name = unique_name("miles-namespace-gc")
        await cluster_core_v1.create_namespace(
            body=kubernetes_client.V1Namespace(metadata=kubernetes_client.V1ObjectMeta(name=name))
        )
        await cluster_core_v1.delete_namespace(name=name, grace_period_seconds=0)

        async def namespace_is_gone() -> bool:
            listed = await cluster_core_v1.list_namespace()
            return name not in [item.metadata.name for item in listed.items]

        await _wait_until_async(
            namespace_is_gone,
            description="the namespace to be fully removed by the finalizer",
            timeout=_TERMINATION_TIMEOUT,
        )

    async def test_the_selector_scopes_a_list_on_the_cluster(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """The label selector the reflector will rely on filters server-side."""
        labelled = pod_body(name="pod-labelled", cell="cell-env")
        unlabelled = pod_body(name="pod-unlabelled", cell="cell-env")
        unlabelled.metadata.labels = {}
        await cluster_core_v1.create_namespaced_pod(namespace=cluster_namespace, body=labelled)
        await cluster_core_v1.create_namespaced_pod(namespace=cluster_namespace, body=unlabelled)

        listed = await cluster_core_v1.list_namespaced_pod(namespace=cluster_namespace, label_selector=CELL_LABEL)
        assert [pod.metadata.name for pod in listed.items] == ["pod-labelled"]


async def _wait_until_async(
    predicate: Callable[[], Awaitable[bool]], *, description: str, timeout: float, interval: float = 1.0
) -> None:
    deadline = time.monotonic() + timeout
    while not await predicate():
        assert time.monotonic() < deadline, f"timed out after {timeout}s waiting until {description}"
        await asyncio.sleep(interval)
