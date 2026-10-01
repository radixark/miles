from __future__ import annotations

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_spec import BaseSpec


def compute_helm_backend_capability(*, specs: list[BaseSpec], config: StaticConnConfig) -> KubernetesBackendCapability:
    release = env.current_release()
    run = KubernetesRunInfo(
        namespace=env.current_namespace(),
        label_selector=f"{env.INSTANCE_LABEL}={release}",
        label_keys=env.DEFAULT_LABEL_KEYS,
        specs={spec.name: spec for spec in specs if spec.scheduling.declares_dynamic_pool()},
    )

    return KubernetesBackendCapability(
        run=run,
        release=release,
        config=config,
        cell_operations=KubernetesCellOperations(
            provider=KubernetesWorkerProvider(
                run=run, pool_ids=sorted(run.specs), resync_period=DEFAULT_RESYNC_PERIOD
            ),
            namespace=run.namespace,
        ),
    )
