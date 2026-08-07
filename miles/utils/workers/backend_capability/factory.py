from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import (
    current_label_keys,
    current_namespace,
    current_release,
)
from miles.utils.workers.worker_spec import BaseWorkerSpec


def get_backend_capability(*, specs: list[BaseWorkerSpec], cluster_backend: ClusterBackend) -> BackendCapability:
    match cluster_backend:
        case ClusterBackend.KUBERNETES:
            return compute_capability(
                specs=specs,
                namespace=current_namespace(),
                release=current_release(),
                label_keys=current_label_keys(),
            )
        case ClusterBackend.RAY:
            return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
