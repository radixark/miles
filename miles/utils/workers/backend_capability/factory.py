from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_spec import BaseSpec


def get_backend_capability(
    *,
    specs: list[BaseSpec],
    cluster_backend: ClusterBackend,
    static_connections: StaticConnConfig,
) -> BackendCapability:
    match cluster_backend:
        case ClusterBackend.KUBERNETES:
            return compute_helm_backend_capability(specs=specs, config=static_connections)
        case ClusterBackend.RAY:
            return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
