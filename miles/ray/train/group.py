import asyncio
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.ray.train.cell import RayTrainCell
from miles.ray.train.cell_monitor import create_trainer_cell_health_checker
from miles.utils.async_utils import AsyncioGatherUtils
from miles.utils.event_analyzer import analyzer as event_analyzer
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent, WitnessAllocateIdEvent
from miles.utils.health_checker import NoopHealthChecker, SimpleHealthCheckerConfig
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.retry_utils import retry
from miles.utils.test_utils.ft_test_actions import FTTestActionGroupExecutor
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

        health_checker_config = (
            SimpleHealthCheckerConfig.from_args(args, prefix="trainer_heartbeat_checker") if num_cells > 1 else None
        )

        def _create_cell(cell_index: int):
            cell_pg = _slice_pg(pg, start=cell_index * gpus_per_cell, end=(cell_index + 1) * gpus_per_cell)

            cell = RayTrainCell(
                args=args,
                role=role,
                with_ref=with_ref,
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

            if health_checker_config is not None:
                cell.health_checker = create_trainer_cell_health_checker(
                    cell=cell,
                    config=health_checker_config,
                    max_heartbeat_age=args.trainer_heartbeat_checker_max_heartbeat_age,
                )

            return cell

        self._cells: list[RayTrainCell] = [_create_cell(cell_index) for cell_index in range(num_cells)]

        self._witness_allocator: WitnessIdAllocator | None = (
            WitnessIdAllocator(buffer_size=args.witness_buffer_size) if args.enable_witness else None
        )
        if self._witness_allocator is not None and args.save_debug_event_data is not None:
            self._witness_allocator.resume(read_persisted_witness_counter(Path(args.save_debug_event_data)))

        self._test_action_executor = FTTestActionGroupExecutor.from_args(args, group=self)

    # ------------------------ API :: train ------------------------

    async def train(self, rollout_id: int, rollout_data_pack):
        """Do one rollout training"""

        event_analyzer.run_analysis_from_args(self.args)

        async def _fn(attempt: int):
            witness_info = self._allocate_witness_info(
                rollout_id=rollout_id,
                attempt=attempt,
                sample_indices=rollout_data_pack["sample_indices"],
            )

            await self._refresh_cells(rollout_id=rollout_id)
            snapshot_alive_cells, results = await self._execute_all_alive_and_catch(
                "train",
                rollout_id=rollout_id,
                rollout_data_ref=rollout_data_pack["data_ref"],
                witness_info=witness_info,
                attempt=attempt,
            )
            self._check_train_one_attempt(results)

            self._log_step_end_event(
                rollout_id=rollout_id,
                snapshot_alive_cells=snapshot_alive_cells,
                results=results,
            )

        await retry(_fn)

        self._test_action_executor.run_after_step(rollout_id=rollout_id)

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

    def _log_step_end_event(self, *, rollout_id: int, snapshot_alive_cells: list, results: list):
        if is_event_logger_initialized():
            cell_outcomes = {
                cell.cell_index: ("error" if isinstance(cell_results, BaseException) else [r for r in cell_results])
                for cell, cell_results in zip(snapshot_alive_cells, results, strict=True)
            }
            get_event_logger().log(
                TrainGroupStepEndEvent,
                dict(rollout_id=rollout_id, cell_outcomes=cell_outcomes),
            )

    @staticmethod
    def _check_train_one_attempt(results):
        non_error_results = [r for r in results if not isinstance(r, BaseException)]
        if not non_error_results:
            raise RuntimeError("All cells failed in this training attempt")

        # NOTE: If some cells errors + all other cells claim normal, we do *not* retry
        #       This may happen when some cells fails *after* exchanging gradients w/ others
        if any(
            any(r == TrainStepOutcome.DISCARDED_SHOULD_RETRY for r in cell_results)
            for cell_results in non_error_results
        ):
            raise ValueError("Exists DISCARDED_SHOULD_RETRY, thus need retry")

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
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        await retry(lambda _: self._execute_first_alive("save_model", rollout_id, force_sync=force_sync))

    async def update_weights(self):
        """Broadcast weights to rollout engines."""
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Fetch the updatable engines + lock once (like V1 RayActorGroup) so all
        # ranks observe a consistent engine set; the actor releases the lock itself.
        info = await self._rollout_manager.get_updatable_engines_and_lock.remote()
        await self._rollout_manager.health_monitoring_pause.remote()
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        await retry(lambda _: self._execute_first_alive("update_weights", info=info))

    async def onload(self):
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("wake_up")
        for cell in self._cells:
            cell.health_checker.resume()

    async def offload(self):
        for cell in self._cells:
            cell.health_checker.pause()
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("sleep")

    async def clear_memory(self):
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("clear_memory")

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

    def stop_cell(self, cell_index: int) -> None:
        self._cells[cell_index].stop()

    def start_cell(self, cell_index: int) -> None:
        """Mark a stopped cell as pending. Actual startup happens in train()."""
        self._cells[cell_index].mark_as_pending()

    # ------------------------ utils to forward calls to cells ------------------------

    async def _execute_all_alive_and_catch(self, fn_name: str, *args, **kwargs):
        snapshot_alive_cells = [c for c in self._cells if c.is_alive]
        assert snapshot_alive_cells, "No alive cells"
        # NOTE: no timeout here. If a cell hangs, the external FT controller
        # detects stale heartbeat via cell_status(), calls cell.stop() to kill
        # actors, which unblocks this gather with ActorDiedError.
        outputs = await asyncio.gather(
            *[cell.execute(fn_name, *args, **kwargs) for cell in snapshot_alive_cells],
            return_exceptions=True,
        )
        AsyncioGatherUtils.log_error(outputs, debug_name=f"execute_all_alive_and_catch#{fn_name}")
        return snapshot_alive_cells, outputs

    async def _execute_first_alive(self, fn_name: str, *args, **kwargs):
        alive_cells = [c for c in self._cells if c.is_alive]
        assert alive_cells, "No alive cells"
        return await alive_cells[0].execute(fn_name, *args, **kwargs)

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self, *, rollout_id: int) -> None:
        snapshotted_pending_indices = [c.cell_index for c in self._cells if c.is_pending]
        snapshotted_alive_indices = [c.cell_index for c in self._cells if c.is_alive]
        will_alive_indices = sorted(list(set(snapshotted_pending_indices + snapshotted_alive_indices)))
        assert len(snapshotted_alive_indices) > 0, "Cannot recover when all cells are dead"

        # Step 0: Determine whether need to reconfigure
        exists_alive_cell_changed_config = any(
            cell.indep_dp_info.alive_cell_indices != will_alive_indices
            for cell in self._cells
            if cell.cell_index in snapshotted_alive_indices
        )
        exists_pending_cell = len(snapshotted_pending_indices) != 0
        needs_reconfigure = exists_pending_cell or exists_alive_cell_changed_config
        if not needs_reconfigure:
            return

        # Step 1: Kill any errored cells and confirm their processes are gone
        # BEFORE the surviving cells reconfigure. Otherwise, indep_dp NCCL abort hangs.
        await self._kill_errored_cells_and_confirm_dead()

        # Step 2: Bump states
        self._indep_dp_quorum_id += 1

        # Step 3: Allocate pending actors
        # We currently do not consider this phase to have errors (because it does not touch GPUs)
        for c in self._cells:
            if c.cell_index in snapshotted_pending_indices:
                c.allocate_for_pending()

        # Step 4: Cooperatively prepare
        src_cell_index = snapshotted_alive_indices[0]  # TODO make it balanced, and support multi-src-to-one-dst
        src_alive_rank = will_alive_indices.index(src_cell_index)
        ckpt_dst_alive_ranks = [will_alive_indices.index(x) for x in snapshotted_pending_indices]

        with _paused_health_checkers(self._cells):
            coop_prepare_outputs = await asyncio.gather(
                *[
                    (
                        c.prepare_indep_dp_mode_alive(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            send_ckpt_dst_ranks=ckpt_dst_alive_ranks if c.cell_index == src_cell_index else [],
                        )
                        if c.cell_index in snapshotted_alive_indices
                        else c.prepare_indep_dp_mode_healing(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            recv_ckpt_src_rank=src_alive_rank if c.cell_index in snapshotted_pending_indices else None,
                        )
                    )
                    for c in self._cells
                    if c.cell_index in will_alive_indices
                ],
                return_exceptions=True,
            )
        # No need to do anything else - cells with exceptions will auto mark itself as errored
        AsyncioGatherUtils.log_error(coop_prepare_outputs, debug_name="refresh_cells#cooperatively_prepare")

        if not AsyncioGatherUtils.has_error(coop_prepare_outputs):
            assert [c.cell_index for c in self._cells if c.is_alive] == will_alive_indices
            self._log_reconfigure_event(
                rollout_id=rollout_id,
                src_cell_index=src_cell_index if snapshotted_pending_indices else None,
                healed_cell_indices=snapshotted_pending_indices,
                alive_cell_indices_after=will_alive_indices,
            )

    def _log_reconfigure_event(
        self,
        *,
        rollout_id: int,
        src_cell_index: int | None,
        healed_cell_indices: list[int],
        alive_cell_indices_after: list[int],
    ) -> None:
        if is_event_logger_initialized():
            get_event_logger().log(
                CellReconfigureEvent,
                dict(
                    rollout_id=rollout_id,
                    quorum_id=self._indep_dp_quorum_id,
                    src_cell_index=src_cell_index,
                    healed_cell_indices=healed_cell_indices,
                    alive_cell_indices_after=alive_cell_indices_after,
                ),
            )

    async def _kill_errored_cells_and_confirm_dead(self) -> None:
        errored_cells = [c for c in self._cells if c.is_errored]
        if not errored_cells:
            return

        logger.info(f"Killing errored cells before reconfigure: {[c.cell_index for c in errored_cells]}")
        await asyncio.gather(*[c.stop_and_confirm_dead() for c in errored_cells])

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


@contextmanager
def _paused_health_checkers(cells: Sequence[RayTrainCell]) -> Iterator[None]:
    for c in cells:
        c.health_checker.pause()
    try:
        yield
    finally:
        for c in cells:
            c.health_checker.resume()
