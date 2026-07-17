# FROZEN: v1 RayTrainGroup is the non-FT default path. Only critical bugfixes
# go here; new features land in miles/ray/train/group.py (v2). Dispatch between
# v1 and v2 happens in miles/ray/placement_group.py based on the env var
# MILES_EXPERIMENTAL_FT_TRAINER (default off -> v1).

import asyncio

from ray.util.placement_group import PlacementGroup

from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.utils.ft_utils.indep_dp import IndepDPInfo


class RayTrainGroup:
    """
    A group of ray actors

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        rollout_manager: object | None,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref
        self._rollout_manager = rollout_manager
        self.with_opd_teacher = with_opd_teacher

        # Allocate the GPUs for actors w/o instantiating them
        self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        return allocate_gpus_for_actor(
            args=self.args,
            gpus_per_cell=self._num_nodes * self._num_gpus_per_node,
            pg=pg,
            num_gpus_per_actor=num_gpus_per_actor,
            indep_dp_store_addr=None,
            role=self.role,
            cell_index=0,
        )

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        indep_dp_info = IndepDPInfo.create_trivial()
        return await self._broadcast(
            "init",
            self.args,
            self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=indep_dp_info,
        )

    async def train(self, rollout_id, rollout_data_pack):
        """Do one rollout training"""
        await self._broadcast(
            "train",
            rollout_id,
            rollout_data_pack["data_ref"],
            witness_info=None,
            attempt=0,
        )

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights from rank 0 to all other ranks."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            await self.rollout_manager.recover_updatable_engines.remote()

        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        await self.rollout_manager.health_monitoring_pause.remote()

        await self._broadcast("update_weights", info=info)

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self):
        self.rollout_manager = self._rollout_manager
        await self._broadcast("set_rollout_manager", self._rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
