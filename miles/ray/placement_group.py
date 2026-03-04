from __future__ import annotations

import asyncio
import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.actor_group import RayTrainGroup
from miles.ray.rollout import RolloutManager
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager

logger = logging.getLogger(__name__)

_MAX_PLACEMENT_RETRIES = 3


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _sort_key(x: tuple[int, str, str]) -> tuple[list[int], str]:
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


def _get_excluded_node_ids() -> set[str]:
    """Query K8s for bad nodes and return a set of all identifiers (hostname + IP).

    Returns both hostname and resolved IP so the exclusion check works regardless
    of whether Ray reports node IPs or hostnames.
    Falls back gracefully on any error (returns empty set, logs warning).
    """
    try:
        manager = K8sNodeManager()
        bad_nodes: list[str] = asyncio.run(manager.get_bad_nodes())
    except Exception:
        logger.warning(
            "Failed to query K8s bad nodes, proceeding without exclusion",
            exc_info=True,
        )
        return set()

    ids: set[str] = set()
    for hostname in bad_nodes:
        ids.add(hostname)
        try:
            ids.add(socket.gethostbyname(hostname))
        except (socket.gaierror, OSError):
            pass
    return ids


def _check_placement_has_excluded_nodes(
    gpu_ids: list[tuple[str, str]],
    excluded: set[str],
) -> set[str]:
    """Return the subset of assigned node identifiers that are in *excluded*.

    ``gpu_ids`` is a list of (node_ip_or_hostname, gpu_id) tuples returned by
    InfoActor — one per bundle.
    """
    assigned_nodes = {node_id for node_id, _gpu_id in gpu_ids}
    return assigned_nodes & excluded


def _create_placement_group_once(num_gpus: int):
    """Create a single placement group and return (pg, reordered_indices, reordered_gpu_ids, gpu_ids)."""
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

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=_sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids, gpu_ids


def _create_placement_group(
    num_gpus: int,
    excluded_node_ids: set[str] | None = None,
):
    """Create a placement group, optionally retrying if bundles land on excluded nodes."""
    excluded = excluded_node_ids or set()

    for attempt in range(_MAX_PLACEMENT_RETRIES + 1):
        pg, reordered_indices, reordered_gpu_ids, gpu_ids = _create_placement_group_once(num_gpus)

        if not excluded:
            return pg, reordered_indices, reordered_gpu_ids

        bad_assigned = _check_placement_has_excluded_nodes(
            gpu_ids=gpu_ids,
            excluded=excluded,
        )
        if not bad_assigned:
            return pg, reordered_indices, reordered_gpu_ids

        logger.warning(
            "Placement group has bundles on excluded nodes %s, retry %d/%d",
            bad_assigned, attempt + 1, _MAX_PLACEMENT_RETRIES,
        )
        ray.util.remove_placement_group(pg)

    raise RuntimeError(
        f"Cannot create placement group avoiding excluded nodes {excluded} "
        f"after {_MAX_PLACEMENT_RETRIES} retries"
    )


def create_placement_groups(args):
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

    excluded: set[str] = set()
    if args.use_fault_tolerance:
        excluded = _get_excluded_node_ids()
        if excluded:
            logger.info("Excluding bad nodes from placement: %s", excluded)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(
        num_gpus=num_gpus,
        excluded_node_ids=excluded,
    )

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    if args.use_critic:
        critic_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[critic_offset:]
        critic_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[critic_offset:]

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
