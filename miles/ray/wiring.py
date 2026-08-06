from __future__ import annotations

from miles.ray.placement_group import create_placement_groups
from miles.ray.specs.entrypoint import compute_specs
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


def launch_worker_manager(args):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)


def get_backend_capability(args) -> BackendCapability:
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            specs = compute_specs(args)
            return compute_capability(
                specs=specs,
                namespace=current_namespace(),
                release=current_release(),
                label_keys=current_label_keys(),
            )
        case ClusterBackend.RAY:
            return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


def _launch_ray_worker_manager(args):
    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
