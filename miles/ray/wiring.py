from miles.ray.placement_group import create_placement_groups
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def launch_worker_manager(args):
    # TODO: after k8s native mode is created, early return when in that mode
    return _launch_ray_worker_manager(args)


def _launch_ray_worker_manager(args):
    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    handle = RayWorkerManager.launch(specs, pgs)
    # TODO: forbid code outside RayWorkerManager to know pg
    return handle, pgs
