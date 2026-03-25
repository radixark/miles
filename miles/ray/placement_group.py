from __future__ import annotations

import logging
from pathlib import Path

import ray

from miles.utils.placement_group_utils import PlacementGroupSlice, create_placement_group_info

from ..utils.control_server_utils import start_control_server
from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


def create_placement_groups(args) -> dict[str, PlacementGroupSlice | None]:
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
            rollout_offset += args.critic_num_nodes * args.critic_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    snapshot_path = Path(args.placement_persist_path) if getattr(args, "placement_persist_path", None) else None
    pg_info = create_placement_group_info(num_gpus, snapshot_path=snapshot_path)

    return {
        "actor": pg_info[:],
        "critic": pg_info[critic_offset:] if args.use_critic else None,
        "rollout": pg_info[rollout_offset:],
    }


def allocate_train_group(args, num_nodes: int, num_gpus_per_node: int, pg: PlacementGroupSlice) -> RayTrainGroup:
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )
    if args.use_critic:
        critic_model = allocate_train_group(
            args=args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
        )
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)
    else:
        critic_model = None

    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.use_critic:
        ray.get(critic_init_handle)
        actor_model.connect(critic_model)

    actor_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    if args.use_control_server:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
        )

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    options_kwargs: dict = dict(num_cpus=1, num_gpus=0)

    if getattr(args, "use_fault_tolerance", False):
        from miles.utils.ft.factories.scheduling import get_cpu_only_scheduling_options

        options_kwargs.update(get_cpu_only_scheduling_options())

    rollout_manager = RolloutManager.options(**options_kwargs).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
