import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

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


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    logger.info("Waiting for placement group to be ready...")
    ray.get(pg.ready())
    logger.info("Placement group is ready.")
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

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    # When not colocating, use separate placement groups to avoid bundle overlap/deadlock.
    if not args.colocate and not args.debug_train_only and not args.debug_rollout_only:
        logger.info("Creating placement groups (separate actor/rollout)...")
        actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_gpus = args.rollout_num_gpus
        actor_pg = _create_placement_group(actor_gpus) if actor_gpus > 0 else None
        rollout_pg = _create_placement_group(rollout_gpus) if rollout_gpus > 0 else None
        if args.use_critic:
            critic_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_pg = _create_placement_group(critic_gpus) if critic_gpus > 0 else None
        else:
            critic_pg = None
        return {
            "actor": actor_pg,
            "critic": critic_pg,
            "rollout": rollout_pg,
        }

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
    logger.info(
        "Placement group offsets: rollout_offset=%s, critic_offset=%s",
        rollout_offset,
        critic_offset if args.use_critic else None,
    )
    pg, all_reordered_bundle_indices, all_reordered_gpu_ids = _create_placement_group(num_gpus)

    def _subset_by_range(start: int, count: int):
        if count <= 0:
            return [], []
        valid = set(range(start, start + count))
        subset_indices = []
        subset_gpu_ids = []
        for bundle_idx, gpu_id in zip(all_reordered_bundle_indices, all_reordered_gpu_ids):
            if bundle_idx in valid:
                subset_indices.append(bundle_idx)
                subset_gpu_ids.append(gpu_id)
        return subset_indices, subset_gpu_ids

    # When colocated, all roles share the full ordered bundle list.
    if args.colocate or args.debug_rollout_only or args.debug_train_only:
        actor_pg_reordered_bundle_indices = all_reordered_bundle_indices
        actor_pg_reordered_gpu_ids = all_reordered_gpu_ids
        rollout_pg_reordered_bundle_indices = all_reordered_bundle_indices if not args.debug_train_only else []
        rollout_pg_reordered_gpu_ids = all_reordered_gpu_ids if not args.debug_train_only else []
        if args.use_critic:
            critic_pg_reordered_bundle_indices = all_reordered_bundle_indices
            critic_pg_reordered_gpu_ids = all_reordered_gpu_ids
    else:
        actor_count = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_count = args.rollout_num_gpus
        actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _subset_by_range(0, actor_count)
        rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids = _subset_by_range(
            rollout_offset, rollout_count
        )
        if args.use_critic:
            critic_count = args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids = _subset_by_range(
                critic_offset, critic_count
            )

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "critic": (pg, critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids) if args.use_critic else None,
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        # Diffusion training is GPU-heavy; avoid fractional-GPU scheduling stalls.
        num_gpus_per_actor=1.0 if args.diffusion_train else 0.4,
    )


def create_training_models(args, pgs, rollout_manager):
    logger.info("Initializing actor/critic models...")
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

    logger.info("Initializing actor model...")
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    logger.info("Actor model initialized.")

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.use_critic:
        ray.get(critic_init_handle)
        actor_model.connect(critic_model)
        logger.info("Critic model initialized and connected.")

    actor_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    use_diffusion_rollout = "diffusion_rollout" in args.rollout_function_path
    logger.info(
        "Creating rollout manager (diffusion=%s, num_gpus=%s)",
        use_diffusion_rollout,
        0 if (use_diffusion_rollout and getattr(args, "rollout_num_gpus", 1) > 1) else (1 if use_diffusion_rollout else 0),
    )
    scheduling_strategy = None
    if use_diffusion_rollout:
        pg_tuple = pg
        # If rollout uses multiple GPUs, do NOT bind RolloutManager to the rollout PG.
        # Otherwise it consumes a GPU bundle and starves rollout workers.
        if getattr(args, "rollout_num_gpus", 1) <= 1:
            pg, reordered_bundle_indices, _ = pg_tuple
            bundle_index = reordered_bundle_indices[0] if reordered_bundle_indices else 0
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )

    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0 if (use_diffusion_rollout and getattr(args, "rollout_num_gpus", 1) > 1) else (1 if use_diffusion_rollout else 0),
        scheduling_strategy=scheduling_strategy,
    ).remote(args, pg_tuple if use_diffusion_rollout else pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        logger.info("Fetching num_rollout_per_epoch from rollout manager...")
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0
        logger.info("Computed num_rollout=%s (num_rollout_per_epoch=%s)", args.num_rollout, num_rollout_per_epoch)

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    logger.info("Rollout manager created.")
    return rollout_manager, num_rollout_per_epoch
