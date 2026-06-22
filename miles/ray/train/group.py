import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup

from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.ray.train.cell import RayTrainCell
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import WitnessAllocateIdEvent
from miles.utils.health_checker import NoopHealthChecker
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.structured_log import log_structured
from miles.utils.witness.allocator import WitnessIdAllocator, read_persisted_witness_counter

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


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
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes: int,
        num_gpus_per_node: int,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        rollout_manager: object | None,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._rollout_manager = rollout_manager

        total_gpus = num_nodes * num_gpus_per_node
        num_cells = (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1
        gpus_per_cell = total_gpus // num_cells
        assert total_gpus % num_cells == 0, f"total_gpus ({total_gpus}) must be divisible by num_cells ({num_cells})"

        self._indep_dp_quorum_id = 0

        if num_cells > 1:
            self._indep_dp_store, indep_dp_store_addr = _create_tcp_store()
            logger.info(f"Created TCPStore for independent DP at {indep_dp_store_addr}")
        else:
            self._indep_dp_store, indep_dp_store_addr = None, None

        def _create_cell(cell_index: int):
            cell_pg = _slice_pg(pg, start=cell_index * gpus_per_cell, end=(cell_index + 1) * gpus_per_cell)

            cell = RayTrainCell(
                args=args,
                role=role,
                with_ref=with_ref,
                with_opd_teacher=with_opd_teacher,
                cell_index=cell_index,
                rollout_manager=rollout_manager,
                actor_factory=lambda _pg=cell_pg, _ci=cell_index: allocate_gpus_for_actor(
                    args=args,
                    gpus_per_cell=gpus_per_cell,
                    pg=_pg,
                    num_gpus_per_actor=num_gpus_per_actor,
                    indep_dp_store_addr=indep_dp_store_addr,
                    role=role,
                    cell_index=_ci,
                ),
                health_checker=NoopHealthChecker(),
            )

            return cell

        self._cells: list[RayTrainCell] = [_create_cell(cell_index) for cell_index in range(num_cells)]

        self._witness_allocator: WitnessIdAllocator | None = (
            WitnessIdAllocator(buffer_size=args.witness_buffer_size) if args.enable_witness else None
        )
        if self._witness_allocator is not None and args.save_debug_event_data is not None:
            self._witness_allocator.resume(read_persisted_witness_counter(Path(args.save_debug_event_data)))

    # ------------------------ API :: train ------------------------

    async def train(self, rollout_id: int, rollout_data_pack):
        """Do one rollout training"""
        witness_info = self._allocate_witness_info(
            rollout_id=rollout_id,
            attempt=0,
            sample_indices=rollout_data_pack["sample_indices"],
        )

        log_structured(logger.info, op="train", phase="start", rollout=rollout_id, attempt=0)
        await self._execute_all(
            "train",
            rollout_id=rollout_id,
            rollout_data_ref=rollout_data_pack["data_ref"],
            witness_info=witness_info,
            attempt=0,
        )

    def _allocate_witness_info(self, *, rollout_id: int, attempt: int, sample_indices):
        if self._witness_allocator is None:
            return None

        witness_info = self._witness_allocator.allocate(num_ids=len(sample_indices))

        if is_event_logger_initialized():
            get_event_logger().log(
                WitnessAllocateIdEvent,
                dict(
                    rollout_id=rollout_id,
                    attempt=attempt,
                    witness_id_to_sample_index=dict(zip(witness_info.witness_ids, sample_indices, strict=True)),
                    counter_after=self._witness_allocator.counter,
                ),
            )

        return witness_info

    # ------------------------ API :: others ------------------------

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        cell_results = await asyncio.gather(
            *[
                cell.init(
                    indep_dp_info=self._compute_indep_dp_info(
                        cell_index=cell.cell_index,
                        # all cells will be alive for this first initialization
                        alive_cell_indices=list(range(len(self._cells))),
                    )
                )
                for cell in self._cells
            ]
        )
        return [item for sublist in cell_results for item in sublist]

    async def save_model(self, rollout_id: int, force_sync: bool = False):
        """Save actor model. Only cell 0 saves to avoid file write conflicts."""
        await self._execute_first("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights to rollout engines."""
        log_structured(logger.info, op="update_weights", phase="start", rollout=rollout_id)
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Fetch the updatable engines + lock once (like V1 RayActorGroup) so all
        # ranks observe a consistent engine set; the actor releases the lock itself.
        info = await self._rollout_manager.get_updatable_engines_and_lock.remote()
        await self._rollout_manager.health_monitoring_pause.remote()
        await self._execute_first("update_weights", info=info)

    async def onload(self):
        await self._execute_all("wake_up")
        for cell in self._cells:
            cell.health_checker.resume()

    async def offload(self):
        for cell in self._cells:
            cell.health_checker.pause()
        await self._execute_all("sleep")

    async def clear_memory(self):
        await self._execute_all("clear_memory")

    async def connect(self, critic_group: "RayTrainGroup"):
        assert len(self._cells) == len(critic_group._cells), (
            f"Actor and critic must have the same number of cells: "
            f"actor has {len(self._cells)}, critic has {len(critic_group._cells)}"
        )
        await asyncio.gather(
            *[
                cell.connect_actor_critic(critic_cell)
                for cell, critic_cell in zip(self._cells, critic_group._cells, strict=True)
            ]
        )

    async def set_rollout_manager(self):
        await asyncio.gather(*[cell.set_rollout_manager() for cell in self._cells])

    # ------------------------ utils to forward calls to cells ------------------------

    async def _execute_all(self, fn_name: str, *args, **kwargs):
        return await asyncio.gather(*[cell.execute(fn_name, *args, **kwargs) for cell in self._cells])

    async def _execute_first(self, fn_name: str, *args, **kwargs):
        return await self._cells[0].execute(fn_name, *args, **kwargs)

    def _compute_indep_dp_info(self, cell_index: int, alive_cell_indices: list[int]) -> IndepDPInfo:
        return IndepDPInfo(
            cell_index=cell_index,
            num_cells=len(self._cells),
            alive_rank=alive_cell_indices.index(cell_index),
            alive_size=len(alive_cell_indices),
            quorum_id=self._indep_dp_quorum_id,
            alive_cell_indices=alive_cell_indices,
        )

    # ------------------------ misc states and utils ------------------------

    @property
    def num_cells(self) -> int:
        return len(self._cells)


PGTuple = tuple[PlacementGroup, list[int], list[int]]


def _slice_pg(pg: PGTuple, start: int, end: int) -> PGTuple:
    placement_group, bundle_indices, gpu_ids = pg
    return placement_group, bundle_indices[start:end], gpu_ids[start:end]


def _create_tcp_store() -> tuple["torch.distributed.TCPStore", str]:
    import torch.distributed

    store = torch.distributed.TCPStore(
        host_name="0.0.0.0",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    host = ray.util.get_node_ip_address()
    port = store.port
    return store, f"{host}:{port}"
