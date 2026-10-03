from __future__ import annotations

from collections.abc import Sequence

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider


class KubernetesBackendCapability(BackendCapability):
    def __init__(
        self,
        *,
        run: KubernetesRunInfo,
        release: str,
        config: StaticConnConfig,
        cell_operations: BaseCellOperations,
    ) -> None:
        self._run = run
        self._release = release
        self._static_conn_infos = config.static_conn_infos
        self._cell_operations = cell_operations

    def dynamic_worker_provider(
        self, *, pool_ids: Sequence[str] | None, category: str | None = None
    ) -> BaseWorkerProvider:
        return KubernetesWorkerProvider(
            run=self._run,
            pool_ids=list(pool_ids) if pool_ids is not None else None,
            category=category,
            resync_period=DEFAULT_RESYNC_PERIOD,
        )

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        config = self._static_conn_infos.get(pool_id)
        assert (
            config is not None
        ), f"{pool_id} is not a static pool of this run, which addresses {sorted(self._static_conn_infos)} statically"
        return StaticWorkerProvider.of_release(release=self._release, config=config)

    def cell_operations(self) -> BaseCellOperations:
        return self._cell_operations
