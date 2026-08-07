# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from kubernetes_asyncio import client as kubernetes_client
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.reflector_utils import pod_names_of, running_reconcile_loop
from tests.e2e.k8s_apiserver.utils import BUSYBOX_IMAGE, CELL_LABEL, pod_body, wait_until

from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodWatchEvent
from miles.utils.workers.reconcile.loop import ReconcileLoop

register_cpu_ci(est_time=660, suite="stage-b-cpu", labels=[])

_BOOKMARK_TIMEOUT = 240.0
_STARTUP_TIMEOUT = 180.0
_RESTART_TIMEOUT = 240.0
_GRACE_PERIOD_SECONDS = 120


class TestReconcileLoopAgainstACluster:
    async def test_the_loop_sees_a_cell_once_its_pods_are_actually_running(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """A cell converges to every member reported Running by a real kubelet."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        for index in range(3):
            await cluster_core_v1.create_namespaced_pod(
                namespace=cluster_namespace, body=pod_body(name=f"pod-{index}", cell="cell-a")
            )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: _running_pod_names(running.loop, "cell-a") == ["pod-0", "pod-1", "pod-2"],
                description="every pod of the cell to be running",
                timeout=_STARTUP_TIMEOUT,
            )
            assert running.reconciles.count("cell-a") >= 1, "kubelet-driven status changes must reach reconcile"

    async def test_deleting_a_pod_shrinks_the_cell(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """A kubelet-confirmed deletion removes exactly that member."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        for index in range(2):
            await cluster_core_v1.create_namespaced_pod(
                namespace=cluster_namespace, body=pod_body(name=f"pod-{index}", cell="cell-a")
            )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: _running_pod_names(running.loop, "cell-a") == ["pod-0", "pod-1"],
                description="both pods to be running",
                timeout=_STARTUP_TIMEOUT,
            )

            await cluster_core_v1.delete_namespaced_pod(
                namespace=cluster_namespace, name="pod-0", grace_period_seconds=0
            )
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-1"],
                description="the deleted pod to leave the cell",
            )

    async def test_a_crashing_pod_is_observed_restarting(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """Restarts a kubelet performs on its own arrive over the watch, not via the initial list."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-crashing",
                cell="cell-a",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", "exit 1"],
                restart_policy="Always",
            ),
        )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-crashing"],
                description="the crashing pod to be observed at all",
                timeout=_STARTUP_TIMEOUT,
            )
            observed_at_sync = _restart_count(running.loop, "cell-a")

            await wait_until(
                lambda: _restart_count(running.loop, "cell-a") > observed_at_sync,
                description="a restart that happened after the initial sync",
                timeout=_RESTART_TIMEOUT,
                interval=1.0,
            )

    async def test_a_pod_terminating_gracefully_stays_in_the_cell_until_it_is_gone(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """A pod under deletion is still a member: only the DELETED event may remove it."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-stubborn",
                cell="cell-a",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", "trap '' TERM; sleep 3600"],
                grace_period_seconds=_GRACE_PERIOD_SECONDS,
            ),
        )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: _running_pod_names(running.loop, "cell-a") == ["pod-stubborn"],
                description="the pod to be running",
                timeout=_STARTUP_TIMEOUT,
            )

            await cluster_core_v1.delete_namespaced_pod(namespace=cluster_namespace, name="pod-stubborn")
            await wait_until(
                lambda: _deletion_timestamps(running.loop, "cell-a") == [True],
                description="the pod to be marked for deletion while still a member",
            )

            await cluster_core_v1.delete_namespaced_pod(
                namespace=cluster_namespace, name="pod-stubborn", grace_period_seconds=0
            )
            await wait_until(
                lambda: running.loop.get_by_parent("cell-a") == [],
                description="the cell to empty once the pod is really gone",
            )

    async def test_a_pod_that_exits_on_its_own_stays_in_the_cell_as_succeeded(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """Reaching a terminal phase is a status change, not a departure from the cell."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-transient",
                cell="cell-a",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", "sleep 5"],
            ),
        )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: _phases(running.loop, "cell-a") == ["Succeeded"],
                description="the pod to finish on its own",
                timeout=_STARTUP_TIMEOUT,
                interval=1.0,
            )
            assert pod_names_of(running.loop, "cell-a") == ["pod-transient"]

    async def test_deleting_every_pod_empties_the_cell(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """A cell that loses its last member is reported empty rather than stale."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace, body=pod_body(name="pod-0", cell="cell-a")
        )

        async with running_reconcile_loop(kube_client=api, namespace=cluster_namespace) as running:
            await wait_until(
                lambda: _running_pod_names(running.loop, "cell-a") == ["pod-0"],
                description="the pod to be running",
                timeout=_STARTUP_TIMEOUT,
            )

            await cluster_core_v1.delete_namespaced_pod(
                namespace=cluster_namespace, name="pod-0", grace_period_seconds=0
            )
            await wait_until(
                lambda: running.loop.get_by_parent("cell-a") == [],
                description="the cell to become empty",
            )


class TestWatchProtocolAgainstACluster:
    async def test_a_bookmark_carries_a_resource_version_the_cursor_can_advance_to(
        self, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """A real BOOKMARK frame carries a resourceVersion the pod API can read off it."""
        api = KubernetesAsyncioPodApi(core_v1_api=cluster_core_v1)
        listed = await api.list_pods(namespace=cluster_namespace, label_selector=CELL_LABEL)

        bookmarks: list[PodWatchEvent] = []
        stream = api.stream_pods(
            namespace=cluster_namespace,
            label_selector=CELL_LABEL,
            resource_version=listed.resource_version,
            timeout_seconds=int(_BOOKMARK_TIMEOUT) + 60,
        )
        collector = asyncio.create_task(_collect_bookmarks(stream, bookmarks))
        try:
            await wait_until(
                lambda: len(bookmarks) > 0,
                description="the cluster to send a bookmark",
                timeout=_BOOKMARK_TIMEOUT,
                interval=1.0,
            )
        finally:
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)
            await stream.aclose()

        assert bookmarks[0].resource_version is not None, f"the cursor would not advance {bookmarks[0]=}"


async def _collect_bookmarks(stream: AsyncGenerator[PodWatchEvent, None], bookmarks: list[PodWatchEvent]) -> None:
    async for event in stream:
        if event.type == "BOOKMARK":
            bookmarks.append(event)


def _running_pod_names(loop: ReconcileLoop, cell: str) -> list[str]:
    return [pod.metadata.name for pod in loop.get_by_parent(cell) if pod.status.phase == "Running"]


def _phases(loop: ReconcileLoop, cell: str) -> list[str]:
    return [pod.status.phase for pod in loop.get_by_parent(cell)]


def _deletion_timestamps(loop: ReconcileLoop, cell: str) -> list[bool]:
    return [pod.metadata.deletion_timestamp is not None for pod in loop.get_by_parent(cell)]


def _restart_count(loop: ReconcileLoop, cell: str) -> int:
    return max((_pod_restart_count(pod) for pod in loop.get_by_parent(cell)), default=0)


def _pod_restart_count(pod: Pod) -> int:
    statuses = pod.status.container_statuses
    return max((status.restart_count for status in statuses), default=0) if statuses else 0
