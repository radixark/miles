# FROZEN: v1 RayTrainGroup is the non-FT default path. Only critical bugfixes
# go here; new features land in miles/ray/train/group.py (v2). Dispatch between
# v1 and v2 happens in miles/ray/placement_group.py based on the env var
# MILES_EXPERIMENTAL_FT_TRAINER (default off -> v1).

import asyncio
from typing import Any

from ray.util.placement_group import PlacementGroup

from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.ray.train_batch_admission import (
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerCellCohort,
    TrainerCohort,
    TrainerRankReceipt,
    validate_data_parallel_coverage,
    validate_publication_data_ref,
)
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
        self._train_batch_pins: dict[TrainBatchPublication, tuple[TrainerAdmissionReceipt, tuple[Any, ...]]] = {}

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

    async def train(self, rollout_id, rollout_data_pack, external_data=None, *, admission_receipt=None):
        """Do one rollout training"""
        rollout_data_ref = rollout_data_pack["data_ref"]
        try:
            if admission_receipt is not None:
                self._validate_train_batch_pin(rollout_id, rollout_data_pack, admission_receipt)
            if external_data is None:
                return await self._broadcast(
                    "train",
                    rollout_id,
                    rollout_data_ref,
                    witness_info=None,
                    attempt=0,
                )
            if isinstance(external_data, list):
                if len(external_data) != len(self._actor_handles):
                    raise ValueError("external_data must contain one payload per train worker")
                refs = [
                    actor.train.remote(
                        rollout_id,
                        rollout_data_ref,
                        witness_info=None,
                        attempt=0,
                        external_data=rank_data,
                    )
                    for actor, rank_data in zip(self._actor_handles, external_data, strict=False)
                ]
                return await asyncio.gather(*refs)
            return await self._broadcast(
                "train",
                rollout_id,
                rollout_data_ref,
                witness_info=None,
                attempt=0,
                external_data=external_data,
            )
        finally:
            if admission_receipt is not None:
                self.discard_train_batch_admission(admission_receipt)

    async def admit_train_batch(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
    ) -> TrainerAdmissionReceipt:
        """Read-proof the exact publication on every v1 trainer rank.

        A rank reads its own shard when the manager splits the publication by
        data-parallel rank. A rank that has not learned its data-parallel layout reads
        every shard. It reads the whole batch when the publication is one reference.
        The group checks that the receipts together cover every published shard.
        """
        publication = rollout_data_pack.get("trainer_admission")
        if not isinstance(publication, TrainBatchPublication) or publication.rollout_id != rollout_id:
            raise ValueError(f"Invalid trainer admission for rollout {rollout_id}.")
        if "data_ref" not in rollout_data_pack:
            raise ValueError(f"Trainer admission for rollout {rollout_id} has no published data reference.")
        data_ref = rollout_data_pack["data_ref"]
        validate_publication_data_ref(publication, data_ref)
        responses = await self._broadcast("admit_train_batch", publication, data_ref)
        expected_ranks = tuple(range(len(self._actor_handles)))
        if len(responses) != len(expected_ranks):
            raise RuntimeError(f"Trainer admission {publication.admission_id} missed a trainer rank response.")
        ranks: list[int] = []
        for response in responses:
            if not isinstance(response, TrainerRankReceipt) or response.publication != publication:
                raise RuntimeError(f"Trainer admission {publication.admission_id} received a stale rank response.")
            if response.rank not in expected_ranks or response.rank in ranks:
                raise RuntimeError(
                    f"Trainer admission {publication.admission_id} received a duplicate or foreign rank."
                )
            ranks.append(response.rank)
        if tuple(sorted(ranks)) != expected_ranks:
            raise RuntimeError(f"Trainer admission {publication.admission_id} missed a trainer rank response.")
        validate_data_parallel_coverage(responses, publication)
        receipt = TrainerAdmissionReceipt(
            publication=publication,
            role=self.role,
            cohort=TrainerCohort(
                quorum_id=None,
                cells=(TrainerCellCohort(cell_index=0, ranks=expected_ranks),),
            ),
        )
        self._train_batch_pins_for_write()[publication] = (receipt, tuple(self._actor_handles))
        return receipt

    def discard_train_batch_admission(self, receipt: TrainerAdmissionReceipt) -> None:
        """Discard the private fixed-rank pin for an admission receipt."""
        pins = self._train_batch_pins_for_write()
        pin = pins.get(receipt.publication)
        if pin is not None and pin[0] == receipt:
            pins.pop(receipt.publication, None)

    def _validate_train_batch_pin(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        receipt: TrainerAdmissionReceipt,
    ) -> None:
        publication = rollout_data_pack.get("trainer_admission")
        if publication != receipt.publication or publication.rollout_id != rollout_id:
            raise RuntimeError("Trainer admission receipt does not match the train batch publication.")
        validate_publication_data_ref(publication, rollout_data_pack["data_ref"])
        pin = self._train_batch_pins_for_write().get(publication)
        if pin is None or pin[0] != receipt:
            raise RuntimeError(f"Trainer admission {publication.admission_id} has no exact trainer pin.")
        expected_handles = pin[1]
        current_handles = tuple(self._actor_handles)
        if len(current_handles) != len(expected_handles) or any(
            current is not expected for current, expected in zip(current_handles, expected_handles, strict=True)
        ):
            raise RuntimeError(f"Trainer admission {publication.admission_id} detected trainer cohort drift.")

    def _train_batch_pins_for_write(self) -> dict:
        if not hasattr(self, "_train_batch_pins"):
            self._train_batch_pins = {}
        return self._train_batch_pins

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def export_hf(self, rollout_id: int, path: str):
        """Export current weights as an HF checkpoint (collective across all ranks)."""
        await self._broadcast("export_hf", rollout_id, path)

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights from rank 0 to all other ranks."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance and "rollout" in self.args.ft_components:
            await self.rollout_manager.recover_updatable_engines.remote()

        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        await self.rollout_manager.health_monitoring_pause.remote()

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

    async def set_rollout_manager(self):
        self.rollout_manager = self._rollout_manager
        await self._broadcast("set_rollout_manager", self._rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
