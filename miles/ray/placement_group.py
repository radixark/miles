import logging
import socket
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_router_providers,
    create_inference_controller_handle,
)
from miles.ray.specs.rollout import create_rollout_executor_handle
from miles.ray.specs.train import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    compute_trainer_configs,
    compute_trainer_ids,
    create_trainer_controller_handle,
    external_trainer_controller_addrs,
)
from miles.ray.wiring import get_backend_capability
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.static import wait_static_addrs_ready

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
    trainer_num_gpus = _compute_trainer_num_gpus(args)
    rollout_num_gpus = args.rollout_num_gpus + args.eval_num_gpus

    if args.debug_train_only:
        return trainer_num_gpus, trainer_num_gpus
    if args.rollout_external:
        return (0, 0) if args.debug_rollout_only else (trainer_num_gpus, trainer_num_gpus)
    if args.debug_rollout_only:
        return rollout_num_gpus, 0
    if args.colocate:
        return max(trainer_num_gpus, rollout_num_gpus), 0
    return trainer_num_gpus + rollout_num_gpus, trainer_num_gpus


def _compute_trainer_num_gpus(args) -> int:
    num_policies = len([config for config in compute_trainer_configs(args) if config.role == ACTOR_ROLE])
    return args.actor_num_nodes * args.actor_num_gpus_per_node * num_policies


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


class TrainerInfo(NamedTuple):
    handle: BaseWorkerHandle
    restored_rollout_id: int
    start_rollout_id: int


# TODO: move (when reorganizing files)
async def create_training_model(args, *, trainer_id: str) -> TrainerInfo:
    handle = create_trainer_controller_handle(args, capability=get_backend_capability(args), trainer_id=trainer_id)
    restored_rollout_ids = await handle.init(args)
    assert len(set(restored_rollout_ids)) == 1, f"trainer {trainer_id!r} restored {restored_rollout_ids}"
    [restored_rollout_id] = set(restored_rollout_ids)
    start_rollout_id = x if (x := args.start_rollout_id) is not None else restored_rollout_id
    return TrainerInfo(handle=handle, restored_rollout_id=restored_rollout_id, start_rollout_id=start_rollout_id)


# TODO: move (when reorganizing files)
async def create_training_models(
    args, rollout_executor: BaseWorkerHandle
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    await wait_external_trainers(args)

    trainer_configs = compute_trainer_configs(args)
    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    actor_info = await create_training_model(
        compute_trainer_args(args, actor_config), trainer_id=actor_config.trainer_id
    )

    critic_configs = [config for config in trainer_configs if config.role == CRITIC_ROLE]
    critic_info = None
    if args.use_critic:
        [critic_config] = critic_configs
        critic_info = await create_training_model(
            compute_trainer_args(args, critic_config), trainer_id=critic_config.trainer_id
        )
        assert critic_info.restored_rollout_id == actor_info.restored_rollout_id, (
            f"the actor restored to rollout {actor_info.restored_rollout_id} but its critic to "
            f"{critic_info.restored_rollout_id}"
        )
    else:
        assert (
            not critic_configs
        ), f"a run without --use-critic needs no critic, but the trainer configs are {trainer_configs}"

    if args.start_rollout_id is None:
        args.start_rollout_id = actor_info.start_rollout_id

    await rollout_executor.set_train_parallel_config(await actor_info.handle.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_info.handle, critic_info.handle if critic_info is not None else None


# TODO: move (when reorganizing files)
async def wait_external_trainers(args) -> None:
    """Wait for every trainer controller another launch deployed, which this run reaches by address."""
    if args.trainer_controller_addrs is None:
        return

    addrs = external_trainer_controller_addrs(args, trainer_ids=compute_trainer_ids(args))
    logger.info(f"Waiting for the independently deployed trainer controllers at {addrs}")
    await wait_static_addrs_ready(addrs.values())


# TODO: move (when reorganizing files)
async def update_weights(
    actor_model, rollout_executor, *, rollout_id: int | None = None, trainer_model_id: str | None = None
) -> None:
    if (weight_version := await actor_model.update_weights(rollout_id=rollout_id)) is not None:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)


# TODO: move (when reorganizing files)
def maybe_start_api_server(
    args, *, trainer_models: dict[str, BaseWorkerHandle], inference_controller: BaseWorkerHandle
) -> None:
    if not args.api_server_port:
        return

    start_api_server(
        args=args,
        trainer_models=trainer_models,
        inference_controller=inference_controller,
        host=args.api_server_host,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=get_backend_capability(args).cell_operations(),
    )


class RolloutComponents(NamedTuple):
    inference_controller: BaseWorkerHandle
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


# TODO: move (when reorganizing files)
async def create_rollout_components(args) -> RolloutComponents:
    capability = get_backend_capability(args)

    if not args.debug_train_only:
        await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))

        session_server_provider = (
            capability.static_worker_provider(pool_id=SESSION_SERVER_POOL_ID) if args.use_session_server else None
        )
        await wait_session_server_ready(args, provider=session_server_provider)

    inference_controller = create_inference_controller_handle(capability=capability)
    await inference_controller.init()

    rollout_executor = create_rollout_executor_handle(capability=capability)
    await rollout_executor.init()

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if (eval_fleet_info := await inference_controller.get_eval_fleet_info()) is not None:
        await rollout_executor.set_eval_fleet_info(eval_fleet_info)

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
