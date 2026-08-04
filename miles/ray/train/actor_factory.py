import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.specs.train import compute_trainer_env_vars
from miles.ray.train_actor import TRAINER_CONCURRENCY_GROUPS
from miles.utils.workers.worker_spec import WorkerLaunchContext


def allocate_gpus_for_actor(
    args,
    gpus_per_cell: int,
    pg: tuple[PlacementGroup, list[int], list[int]],
    num_gpus_per_actor: float,
    indep_dp_store_addr: str,
    role: str,
    cell_index: int,
):
    world_size = gpus_per_cell

    # Use placement group to lock resources for models of same type
    assert pg is not None
    pg, reordered_bundle_indices, _reordered_gpu_ids = pg

    backend = args.train_backend
    if backend == "megatron":
        from miles.backends.megatron_utils.actor import MegatronTrainRayActor

        actor_impl = MegatronTrainRayActor

    else:
        from miles.backends.fsdp_utils import FSDPTrainRayActor

        actor_impl = FSDPTrainRayActor

    TrainRayActor = ray.remote(
        num_gpus=1,
        concurrency_groups=TRAINER_CONCURRENCY_GROUPS,
    )(actor_impl)

    # Create worker actors
    actor_handles = []
    for rank in range(world_size):
        options = dict(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        )
        options["runtime_env"] = {
            "env_vars": compute_trainer_env_vars(
                args,
                WorkerLaunchContext(cell_index=cell_index, worker_in_cell_index=rank, gpu_ids=[]),
            )
        }
        actor = TrainRayActor.options(**options).remote(
            args=args,
            world_size=world_size,
            rank=rank,
            indep_dp_store_addr=indep_dp_store_addr,
            role=role,
            cell_index=cell_index,
        )
        actor_handles.append(actor)

    if actor_handles:
        master_addr, master_port = ray.get(actor_handles[0].propose_master_addr_and_port.remote())
        ray.get(
            [
                actor.configure_master_addr_and_port.remote(master_addr=master_addr, master_port=master_port)
                for actor in actor_handles
            ]
        )

    return actor_handles
