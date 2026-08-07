from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm.labels import DEFAULT_LABEL_KEYS, INSTANCE_LABEL
from miles.utils.workers.worker_provider.kubernetes.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.run import KubernetesRun, kubernetes_run, static_worker_provider
from miles.utils.workers.worker_provider.kubernetes.views.pod_info import CellLabelKeys
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import BaseWorkerSpec


class KubernetesBackendCapability(BackendCapability):
    def __init__(
        self,
        *,
        run: KubernetesRun,
        static_provider: SimpleWorkerProvider,
        cell_operations: BaseCellOperations,
    ) -> None:
        self._run = run
        self._static_provider = static_provider
        self._cell_operations = cell_operations

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        unknown = [name for name in pool_ids if name not in self._run.pools]
        assert not unknown, f"{unknown} are not pool_ids of this run, which deploys {sorted(self._run.pools)}"
        return KubernetesWorkerProvider(run=self._run, pool_ids=list(pool_ids))

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        assert self._static_provider.knows_pool(pool_id), f"{pool_id} is not a static pool of this run"
        return self._static_provider

    def cell_operations(self) -> BaseCellOperations:
        return self._cell_operations


def compute_kubernetes_backend_capability(
    *,
    specs: list[BaseWorkerSpec],
    namespace: str,
    release: str,
    kubernetes_client_factory: Callable[[], Any],
    num_gpus_per_node: int,
    label_keys: CellLabelKeys | None = None,
    colocated_with: Callable[[str], list[str]] | None = None,
) -> KubernetesBackendCapability:
    run = kubernetes_run(
        specs=specs,
        namespace=namespace,
        label_selector=f"{INSTANCE_LABEL}={release}",
        kubernetes_client_factory=kubernetes_client_factory,
        num_gpus_per_node=num_gpus_per_node,
        label_keys=label_keys or DEFAULT_LABEL_KEYS,
    )

    return KubernetesBackendCapability(
        run=run,
        static_provider=static_worker_provider(specs=specs, release=release),
        cell_operations=KubernetesCellOperations(
            provider=KubernetesWorkerProvider(run=run, pool_ids=sorted(run.pools)),
            namespace=namespace,
            colocated_with=colocated_with,
        ),
    )
