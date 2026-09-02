import asyncio
import logging
import socket
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.megatron_utils.checkpoint_tracker import read_checkpoint_tracker_iteration
from miles.backends.megatron_utils.megatron_config import MegatronTrainerConfig, compute_trainer_args
from miles.ray.rollout.inference_controller import UpdatableEngines
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
    TRAINER_CONTROLLER_ADDRS_FLAG,
    compute_trainer_configs,
    create_trainer_controller_handle,
    external_trainer_controller_addrs,
)
from miles.ray.wiring import get_backend_capability
from miles.utils.async_utils import await_task_result_despite_cancel, wait_task_until_done_despite_cancel
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.hot_restart import (
    init_or_reset_inference_controller,
    trainer_init_or_load_state,
    wait_trainers_idle,
    wait_until_worker_not_initialized,
)
from miles.utils.test_utils.ft_test_actions import FTTestActionOrchestrationExecutor
from miles.utils.workers.types import DeployComponent, DeploymentIdentity
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.static import wait_static_addrs_ready

logger = logging.getLogger(__name__)

WEIGHT_UPDATE_TIMEOUT_SECONDS = 600.0


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
    selector = DeployComponent(args.deploy_component)
    trainer_num_gpus = _compute_trainer_num_gpus(args) if selector.selects(DeployComponent.TRAINER) else 0
    if args.debug_train_only:
        return trainer_num_gpus, trainer_num_gpus

    rollout_num_gpus = args.rollout_num_gpus + args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
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
def create_trainer_handles(args, *, trainer_configs: list[MegatronTrainerConfig]) -> dict[str, BaseWorkerHandle]:
    capability = get_backend_capability(args)
    return {
        config.trainer_id: create_trainer_controller_handle(args, capability=capability, trainer_id=config.trainer_id)
        for config in trainer_configs
    }


# TODO: move (when reorganizing files)
async def take_over_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> bool:
    await wait_external_trainers(args, handles=handles)
    resumed = await wait_trainers_idle(handles)

    if resumed and not _trainer_has_checkpoint(args):
        event_logger_checkpoint.discard(args)

    return resumed


def _trainer_has_checkpoint(args) -> bool:
    assert args.megatron_config is None, "a multi policy run's base --load holds no tracker to read"
    return read_checkpoint_tracker_iteration(args.requested_load) is not None


# TODO: move (when reorganizing files)
async def create_training_model(args, *, handle: BaseWorkerHandle, trainer_id: str, resumed: bool) -> TrainerInfo:
    restored_rollout_ids = await trainer_init_or_load_state(handle, args, trainer_id=trainer_id, resumed=resumed)
    assert len(set(restored_rollout_ids)) == 1, f"trainer {trainer_id!r} restored {restored_rollout_ids}"
    [restored_rollout_id] = set(restored_rollout_ids)

    if (x := args.start_rollout_id) is None:
        start_rollout_id = restored_rollout_id
    else:
        if x != restored_rollout_id:
            logger.info(
                f"trainer {trainer_id!r} restored rollout {restored_rollout_id}, and --start-rollout-id {x} was "
                f"asked for, so it starts at {x}"
            )
        start_rollout_id = x

    return TrainerInfo(handle=handle, restored_rollout_id=restored_rollout_id, start_rollout_id=start_rollout_id)


# TODO: move (when reorganizing files)
async def create_training_models(
    args, rollout_executor: BaseWorkerHandle
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    trainer_configs = compute_trainer_configs(args)
    handles = create_trainer_handles(args, trainer_configs=trainer_configs)
    resumed = await take_over_trainers(args, handles=handles)

    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    actor_info = await create_training_model(
        compute_trainer_args(args, actor_config),
        handle=handles[actor_config.trainer_id],
        trainer_id=actor_config.trainer_id,
        resumed=resumed,
    )

    critic_configs = [config for config in trainer_configs if config.role == CRITIC_ROLE]
    critic_info = None
    if args.use_critic:
        [critic_config] = critic_configs
        critic_info = await create_training_model(
            compute_trainer_args(args, critic_config),
            handle=handles[critic_config.trainer_id],
            trainer_id=critic_config.trainer_id,
            resumed=resumed,
        )
        assert critic_info.restored_rollout_id == actor_info.restored_rollout_id, (
            f"the actor restored to rollout {actor_info.restored_rollout_id} but its critic to "
            f"{critic_info.restored_rollout_id}"
        )
    else:
        assert (
            not critic_configs
        ), f"a run without --use-critic needs no critic, but the trainer configs are {trainer_configs}"

    args.start_rollout_id = actor_info.start_rollout_id

    await rollout_executor.set_train_parallel_config(await actor_info.handle.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_info.handle, critic_info.handle if critic_info is not None else None


# TODO: move (when reorganizing files)
async def wait_external_trainers(args, *, handles: dict[str, BaseWorkerHandle]) -> None:
    """Wait for every independently deployed trainer controller, and refuse one that another run deployed."""
    if args.trainer_controller_addrs is None:
        return

    addrs = external_trainer_controller_addrs(args, trainer_ids=list(handles))
    logger.info(f"Waiting for the independently deployed trainer controllers at {addrs}")
    await wait_static_addrs_ready(addrs.values())

    identities = await asyncio.gather(*[handle.get_deployment_identity() for handle in handles.values()])
    for trainer_id, identity in zip(handles, identities, strict=True):
        _assert_external_trainer_belongs_to_this_run(identity, args=args, trainer_id=trainer_id)


def _assert_external_trainer_belongs_to_this_run(
    identity: DeploymentIdentity, *, args, trainer_id: str | None = None
) -> None:
    assert identity.run_uuid == args.run_uuid, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, but this launch drives run {args.run_uuid}: every deployment a split run reaches has "
        f"to be a deployment of that same run, or its weight updates and its rollout samples belong to different runs"
    )
    assert identity.deploy_component == DeployComponent.TRAINER.value, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names the {identity.deploy_component} deployment of run "
        f"{identity.run_uuid}, and only a deployment that carries nothing but the trainer is reached by address: "
        f"an {DeployComponent.ALL.value} release of this run runs an orchestration script of its own, so both "
        f"scripts would drive the same trainer"
    )
    assert identity.deploy_instance_id is None or identity.deploy_instance_id == trainer_id, (
        f"trainer {trainer_id!r} answers as deployment {identity.deploy_instance_id!r}; "
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} entries are keyed by trainer id"
    )


# TODO: move (when reorganizing files)
async def update_weights(
    args,
    actor_model,
    rollout_executor,
    inference_controller: BaseWorkerHandle,
    *,
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
) -> None:
    if rollout_id is not None:
        await FTTestActionOrchestrationExecutor.from_args(args, trainer_model_id=trainer_model_id).run_after_step(
            rollout_id=rollout_id
        )

    info: UpdatableEngines = await inference_controller.start_update_weights(model_id=trainer_model_id)
    update_task = asyncio.create_task(actor_model.update_weights(info=info, rollout_id=rollout_id))
    cancelled = await wait_task_until_done_despite_cancel(update_task, timeout=WEIGHT_UPDATE_TIMEOUT_SECONDS)

    if not update_task.done():
        update_task.cancel()
        try:
            await _end_update_weights(inference_controller, snapshot_cell_id_to_hashes={})
        except Exception:
            logger.exception("Failed to close the inference weight update window after the trainer broadcast hung")
        raise TimeoutError(
            f"The trainer weight broadcast did not finish within {WEIGHT_UPDATE_TIMEOUT_SECONDS}s. Its window holds "
            f"a detached lock that every rollout cell suspension and fault injection waits behind, so the window is "
            f"closed here rather than left open for a broadcast that may never return."
        )

    try:
        weight_version = update_task.result()
    except asyncio.CancelledError:
        await _end_update_weights(inference_controller, snapshot_cell_id_to_hashes={})
        raise
    except Exception:
        try:
            await _end_update_weights(inference_controller, snapshot_cell_id_to_hashes={})
        except Exception:
            logger.exception("Failed to close the inference weight update window after the trainer broadcast failed")
        raise

    if cancelled:
        try:
            await _end_update_weights(inference_controller, snapshot_cell_id_to_hashes={})
        except Exception:
            logger.exception(
                "Failed to close the inference weight update window after the trainer broadcast was cancelled"
            )
        raise asyncio.CancelledError()

    await _end_update_weights(inference_controller, snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

    await _maybe_log_inference_engine_weight_checksums(
        args, inference_controller=inference_controller, rollout_id=rollout_id, trainer_model_id=trainer_model_id
    )

    if weight_version is not None:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)


async def _end_update_weights(
    inference_controller: BaseWorkerHandle, *, snapshot_cell_id_to_hashes: dict[str, str]
) -> None:
    task = asyncio.create_task(
        inference_controller.end_update_weights(snapshot_cell_id_to_hashes=snapshot_cell_id_to_hashes)
    )
    await await_task_result_despite_cancel(task)


async def _maybe_log_inference_engine_weight_checksums(
    args, *, inference_controller: BaseWorkerHandle, rollout_id: int | None, trainer_model_id: str | None
) -> None:
    if not args.save_inference_engine_weight_checksum:
        return
    if not is_event_logger_initialized():
        return
    if args.debug_train_only or args.debug_rollout_only:
        return

    check_weights_result = await inference_controller.check_weights(action="checksum", model_id=trainer_model_id)
    engine_checksums = flatten_inference_engine_checksums(check_weights_result)
    get_event_logger().log(
        InferenceEngineWeightChecksumEvent,
        dict(
            rollout_id=args.start_rollout_id - 1 if rollout_id is None else rollout_id,
            engine_checksums=engine_checksums,
        ),
    )


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

    rollout_executor = create_rollout_executor_handle(capability=capability)
    await wait_until_worker_not_initialized(rollout_executor)

    inference_controller = create_inference_controller_handle(capability=capability)
    await init_or_reset_inference_controller(inference_controller, args=args)

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
