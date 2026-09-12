# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from tests.e2e.k8s_apiserver.utils import CELL_LABEL

from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import KubernetesPodApi, PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop


def pod_cell(pod: Pod) -> str:
    return pod.metadata.labels[CELL_LABEL]


class CountingPodApi:
    def __init__(self, *, inner: KubernetesPodApi) -> None:
        self.list_count = 0
        self.stream_cursors: list[str] = []
        self._inner = inner

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        self.list_count += 1
        return await self._inner.list_pods(namespace=namespace, label_selector=label_selector)

    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        self.stream_cursors.append(resource_version)
        return self._inner.stream_pods(
            namespace=namespace,
            label_selector=label_selector,
            resource_version=resource_version,
            timeout_seconds=timeout_seconds,
        )


class ReconcileRecorder:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.snapshots: dict[str, list[str]] = {}
        self.loop: ReconcileLoop | None = None

    async def __call__(self, key: str) -> None:
        assert self.loop is not None, "the recorder must be bound to a loop before it is driven"
        self.keys.append(key)
        self.snapshots[key] = [pod.metadata.name for pod in self.loop.get_by_parent(key)]

    def count(self, key: str) -> int:
        return self.keys.count(key)


@dataclass(frozen=True)
class RunningLoop:
    loop: ReconcileLoop
    reconciles: ReconcileRecorder


@asynccontextmanager
async def running_reconcile_loop(
    *, kube_client: KubernetesPodApi, namespace: str, watch_timeout_seconds: int = 60
) -> AsyncIterator[RunningLoop]:
    reflector = KubernetesReflector(
        kube_client=kube_client,
        namespace=namespace,
        label_selector=CELL_LABEL,
        watch_timeout_seconds=watch_timeout_seconds,
        retry_delay=0.5,
    )
    recorder = ReconcileRecorder()
    loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell)
    recorder.loop = loop

    await loop.start()
    try:
        yield RunningLoop(loop=loop, reconciles=recorder)
    finally:
        await loop.stop()


def pod_names_of(loop: ReconcileLoop, cell: str) -> list[str]:
    return [pod.metadata.name for pod in loop.get_by_parent(cell)]
