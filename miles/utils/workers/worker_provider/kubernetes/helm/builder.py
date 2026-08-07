from __future__ import annotations

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm.labels import DEFAULT_LABEL_KEYS, INSTANCE_LABEL
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_capability(
    *,
    specs: list[BaseWorkerSpec],
    namespace: str,
    release: str,
    label_keys: CellLabelKeys | None = None,
) -> KubernetesBackendCapability:
    run = KubernetesRunInfo(
        namespace=namespace,
        label_selector=f"{INSTANCE_LABEL}={release}",
        label_keys=label_keys or DEFAULT_LABEL_KEYS,
        specs={spec.name: spec for spec in specs if _declares_dynamic_pool(spec)},
    )

    return KubernetesBackendCapability(
        run=run,
        release=release,
        static_specs={spec.name: spec for spec in specs if not _declares_dynamic_pool(spec)},
        cell_operations=KubernetesCellOperations(
            provider=KubernetesWorkerProvider(run=run, pool_ids=sorted(run.specs)), namespace=namespace
        ),
    )


def _declares_dynamic_pool(spec: BaseWorkerSpec) -> bool:
    return spec.scheduling.gpus_per_cell() > 0
