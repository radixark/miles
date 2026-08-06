from __future__ import annotations

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def launch_worker_manager(args):
    # TODO: after k8s native mode is created, early return when in that mode
    return _launch_ray_worker_manager(args)


def get_backend_capability(args) -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
