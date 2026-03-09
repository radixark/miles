from __future__ import annotations

import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.placement_group_utils import BundleProbe, PlacementGroupInfo, PlacementGroupSlice

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _bundle_sort_key(probe: BundleProbe) -> tuple[list[int], str]:
    node_identifier = probe.node_ip
    try:
        node_ip_parts = list(map(int, node_identifier.split(".")))
    except ValueError:
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Convert each character to its ASCII value for stable sorting
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, probe.gpu_id)


def _create_placement_group(num_gpus: int) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())

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

    probes = [
        BundleProbe(bundle_index=i, node_ip=gpu_ids[i][0], gpu_id=gpu_ids[i][1])
        for i in range(num_bundles)
    ]
    sorted_probes = sorted(probes, key=_bundle_sort_key)

    for rank, probe in enumerate(sorted_probes):
        logger.info(
            f"  bundle {rank:4}, actual_bundle_index: {probe.bundle_index:4}, "
            f"node: {probe.node_ip}, gpu: {probe.gpu_id}"
        )

    return PlacementGroupInfo(pg=pg, bundles=sorted_probes)


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
    pg_info = _create_placement_group(num_gpus)

    actor_count = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_count = num_gpus - rollout_offset

    critic_slice: PlacementGroupSlice | None = None
    if args.use_critic:
        critic_slice = pg_info.slice(
            offset=critic_offset,
            count=args.critic_num_nodes * args.critic_num_gpus_per_node,
        )

    return {
        "actor": pg_info.slice(offset=0, count=actor_count),
        "critic": critic_slice,
        "rollout": pg_info.slice(offset=rollout_offset, count=rollout_count),
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

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

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
