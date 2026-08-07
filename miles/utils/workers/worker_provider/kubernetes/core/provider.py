from __future__ import annotations

import logging
from collections.abc import Awaitable

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.k8s_types import Pod
from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD, ReconcileLoop
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.kubernetes.core import cell_view, pod_view
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys
from miles.utils.workers.worker_spec import BaseWorkerSpec, NamedHostAndPorts, ServeWorkerSpec

logger = logging.getLogger(__name__)


class KubernetesRunInfo(FrozenStrictBaseModel):
    namespace: str
    label_selector: str
    label_keys: CellLabelKeys
    specs: dict[str, BaseWorkerSpec]


class KubernetesWorkerProvider(BaseWorkerProvider):
    def __init__(
        self, *, run: KubernetesRunInfo, pool_ids: list[str], resync_period: float | None = DEFAULT_RESYNC_PERIOD
    ) -> None:
        self._run = run
        self._pool_ids = pool_ids
        self._worker_classes = {
            pool_id: spec.worker_class for pool_id, spec in run.specs.items() if isinstance(spec, ServeWorkerSpec)
        }
        self._resync_period = resync_period
        self._loop: ReconcileLoop | None = None

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        for cell_id in self.cell_ids():
            for worker in self._workers_of_cell(cell_id):
                if worker.name == worker_name:
                    return cell_view.addrs_of_worker(worker, run=self._run)
        raise AssertionError(f"no observed pod serves {worker_name}")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            cell_view.compute_worker_infos(
                cell_id, pods=self._pods_of_cell(cell_id), run=self._run, worker_classes=self._worker_classes
            )
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        reflector = KubernetesReflector(
            kube_client=_create_kubernetes_client(),
            namespace=self._run.namespace,
            label_selector=_watched_pods_selector(
                base_selector=self._run.label_selector,
                pool_label_key=self._run.label_keys.pool_id,
                pool_ids=self._pool_ids,
            ),
        )
        loop = ReconcileLoop(
            source=reflector.watch,
            reconcile=lambda cell_id: self._notify_cell(cell_id, reconcile=reconcile),
            key_map=self._cell_id_of_wanted_pod,
            resync_period=self._resync_period,
        )
        try:
            await loop.start()
        except BaseException:
            await loop.stop()
            raise
        self._loop = loop
        return loop.stop

    async def get_cell_infos(self, *, pool_id: str) -> dict[str, CellInfo]:
        infos = {cell_id: self.cell_info(cell_id) for cell_id in self.cell_ids()}
        return {cell_id: info for cell_id, info in infos.items() if info is not None and info.pool_id == pool_id}

    def cell_ids(self) -> list[str]:
        return sorted(self._loop_or_fail().parent_keys())

    def cell_info(self, cell_id: str) -> CellInfo | None:
        return cell_view.compute_cell_info(cell_id, pods=self._pods_of_cell(cell_id), run=self._run)

    def pod_names(self, cell_id: str) -> list[str]:
        return [pod.name for pod in self._pods_of_cell(cell_id)]

    def _notify_cell(self, cell_id: str, *, reconcile: ReconcileFn) -> Awaitable[None]:
        info = self.cell_info(cell_id)
        return reconcile(cell_id, info if info is not None and info.alive else None)

    def _cell_id_of_wanted_pod(self, pod: Pod) -> str | None:
        parsed = pod_view.parse_pod(pod, self._run.label_keys)
        if parsed is None or parsed.pool_id not in self._pool_ids:
            return None
        return parsed.cell_id

    def _workers_of_cell(self, cell_id: str) -> list[cell_view.RankedWorker]:
        return cell_view.workers_of_pods(self._pods_of_cell(cell_id), run=self._run)

    def _pods_of_cell(self, cell_id: str) -> list[pod_view.ParsedPod]:
        return pod_view.parse_pods(self._loop_or_fail().get_by_parent(cell_id), keys=self._run.label_keys)

    def _loop_or_fail(self) -> ReconcileLoop:
        assert self._loop is not None, "watch_cells must be running before the cells can be read"
        return self._loop


def _create_kubernetes_client() -> KubernetesAsyncioPodApi:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    return KubernetesAsyncioPodApi(core_v1_api=kubernetes_client.CoreV1Api(kubernetes_client.ApiClient()))


def _watched_pods_selector(*, base_selector: str, pool_label_key: str, pool_ids: list[str]) -> str:
    if not pool_ids:
        return f"{base_selector},{_NO_POD_CARRIES_THIS_LABEL}"
    wanted = ",".join(sorted(pool_ids))
    return f"{base_selector},{pool_label_key} in ({wanted})"


_NO_POD_CARRIES_THIS_LABEL = "never-matches.invalid/watches-nothing"
