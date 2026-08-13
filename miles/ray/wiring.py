from __future__ import annotations

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


def launch_worker_manager(args):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)


def get_backend_capability(args) -> BackendCapability:
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))
