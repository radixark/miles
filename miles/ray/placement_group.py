import logging
import socket
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.specs.inference import create_inference_controller_handle
from miles.ray.specs.rollout import create_rollout_executor_handle
from miles.ray.specs.train import compute_critic_args, create_trainer_controller_handle
from miles.utils.workers.worker_handle import BaseWorkerHandle


logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


class PlacementGroupInfo(NamedTuple):
    pg: PlacementGroup
    pg_reordered_bundle_indices: list[int]
    pg_reordered_gpu_ids: list[int]


def _create_placement_group(num_gpus) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return PlacementGroupInfo(pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids)


def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0
    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus
    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0
    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0
    return actor_num_gpus + args.rollout_num_gpus + args.eval_num_gpus, actor_num_gpus


def create_placement_groups(args) -> dict[str, PlacementGroupInfo]:
    """Create placement groups for actor and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    ans = {
        "actor": PlacementGroupInfo(pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": PlacementGroupInfo(pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    if args.use_critic:
        ans["critic"] = ans["actor"]
    return ans


async def create_training_models(args, rollout_executor) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    actor_model = create_trainer_controller_handle(role="actor")
    actor_start_rollout_ids = await actor_model.init(args)

    if args.use_critic:
        critic_model = create_trainer_controller_handle(role="critic")
        critic_start_rollout_ids = await critic_model.init(compute_critic_args(args))
    else:
        critic_model = None

    start_rollout_ids = critic_start_rollout_ids if args.use_critic else actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    await rollout_executor.set_train_parallel_config(await actor_model.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_model, critic_model


async def update_weights(actor_model, rollout_executor, *, rollout_id: int | None = None) -> None:
    if (weight_version := await actor_model.update_weights(rollout_id=rollout_id)) is not None:
        await rollout_executor.set_weight_version(weight_version)


class RolloutComponents(NamedTuple):
    inference_controller: BaseWorkerHandle
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


async def create_rollout_components(args) -> RolloutComponents:
    if not args.debug_train_only:
        await resolve_router_addrs(args)
        await wait_session_server_ready(args)

    inference_controller = create_inference_controller_handle()
    await inference_controller.init()

    rollout_executor = create_rollout_executor_handle()
    await rollout_executor.init()

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    await rollout_executor.set_eval_fleet.remote(inference_controller.eval_fleet)

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
