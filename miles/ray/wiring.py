from miles.ray.placement_group import create_placement_groups
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import specs_trainer
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def launch_worker_manager(args, *, trainer_only: bool = False):
    # TODO: after k8s native mode is created, early return when in that mode
    return _launch_ray_worker_manager(args, trainer_only=trainer_only)


def _launch_ray_worker_manager(args, *, trainer_only: bool = False):
    specs = specs_trainer(args) if trainer_only else compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
