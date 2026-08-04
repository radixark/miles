# FROZEN: v1 RayTrainGroup is the non-FT default path. Only critical bugfixes
# go here; new features land in miles/ray/train/group.py (v2). Dispatch between
# v1 and v2 happens in miles/ray/placement_group.py based on the env var
# MILES_EXPERIMENTAL_FT_TRAINER (default off -> v1).

import asyncio

from ray.util.placement_group import PlacementGroup

from miles.ray.rollout.inference_controller import update_weights_window
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
        inference_controller: object | None,
        rollout_executor: object | None,
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
        self._inference_controller = inference_controller
        self._rollout_executor = rollout_executor
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
        return await self._broadcast(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=IndepDPInfo.create_trivial(),
        )

    async def train(self, rollout_id, rollout_data_pack, external_data=None):
        """Do one rollout training"""
        external_data_kwargs = self._compute_external_data_kwargs(external_data)
        return await self._broadcast_per_worker(
            "train",
            compute_kwargs=lambda worker_index: dict(
                rollout_id=rollout_id,
                rollout_data_ref=rollout_data_pack["data_ref"],
                witness_info=None,
                attempt=0,
                **external_data_kwargs[worker_index],
            ),
        )

    def _compute_external_data_kwargs(self, external_data) -> list[dict]:
        if external_data is None:
            return [{} for _ in self._actor_handles]
        if not isinstance(external_data, list):
            return [dict(external_data=external_data) for _ in self._actor_handles]
        if len(external_data) != len(self._actor_handles):
            raise ValueError("external_data must contain one payload per train worker")
        return [dict(external_data=payload) for payload in external_data]

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id=rollout_id, force_sync=force_sync)

    async def export_hf(self, rollout_id: int, path: str):
        """Export current weights as an HF checkpoint (collective across all ranks)."""
        await self._broadcast("export_hf", rollout_id, path)

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights from rank 0 to all other ranks."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        async with update_weights_window(self._inference_controller) as info:
            await self._broadcast("update_weights", info=info)

    async def reconcile_adapters(self) -> None:
        """Multi-LoRA: reconcile loaded adapters with the controller's active set
        (load new, cleanup gone). Called by the trainer before generate."""
        await self._broadcast("reconcile_adapters")

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def set_rollout_executor(self):
        await self._broadcast("set_rollout_executor", rollout_executor=self._rollout_executor)

    async def _broadcast(self, method_name: str, **kwargs) -> list:
        return await self._broadcast_per_worker(method_name, compute_kwargs=lambda _: kwargs)

    async def _broadcast_per_worker(self, method_name: str, *, compute_kwargs) -> list:
        refs = [
            getattr(actor, method_name).remote(**compute_kwargs(worker_index))
            for worker_index, actor in enumerate(self._actor_handles)
        ]
        return await asyncio.gather(*refs)
