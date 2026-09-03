import asyncio
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
from ray.util.placement_group import PlacementGroup

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.train.actor_factory import allocate_gpus_for_actor
from miles.ray.train.cell import RayTrainCell
from miles.ray.train.cell_monitor import create_trainer_cell_health_checker
from miles.ray.train_batch_admission import (
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerCellCohort,
    TrainerCohort,
    TrainerCohortChangedError,
    TrainerRankReceipt,
    validate_data_parallel_coverage,
    validate_publication_data_ref,
)
from miles.utils.async_utils import AsyncioGatherUtils
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
)
from miles.utils.audit_utils.witness.allocator import WitnessIdAllocator, read_persisted_witness_counter
from miles.utils.ft_utils.health_checker import NoopHealthChecker, SimpleHealthCheckerConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.retry_utils import retry
from miles.utils.test_utils.ft_test_actions import FTTestActionGroupExecutor
from miles.utils.tracking_utils.structured_log import log_structured

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

_RETRY_MAX_ATTEMPTS = 30


def _same_identity_sequence(actual: Sequence[object], expected: Sequence[object]) -> bool:
    return len(actual) == len(expected) and all(
        actual_item is expected_item for actual_item, expected_item in zip(actual, expected, strict=True)
    )


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
        self.role = role

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

            if health_checker_config is not None:
                cell.health_checker = create_trainer_cell_health_checker(
                    cell=cell,
                    config=health_checker_config,
                )

            return cell

        self._cells: list[RayTrainCell] = [_create_cell(cell_index) for cell_index in range(num_cells)]
        self._train_batch_pins: dict[TrainBatchPublication, tuple] = {}

        self._witness_allocator: WitnessIdAllocator | None = (
            WitnessIdAllocator(buffer_size=args.witness_buffer_size) if args.enable_witness else None
        )
        if self._witness_allocator is not None and args.save_debug_event_data is not None:
            self._witness_allocator.resume(read_persisted_witness_counter(Path(args.save_debug_event_data)))

        self._test_action_executor = FTTestActionGroupExecutor.from_args(args, group=self)

    # ------------------------ API :: train ------------------------

    async def train(self, rollout_id: int, rollout_data_pack, *, admission_receipt=None):
        """Do one rollout training"""

        event_analyzer.run_analysis_from_args(self.args)
        cohort_error: TrainerCohortChangedError | None = None

        async def _fn(attempt: int):
            nonlocal cohort_error
            try:
                await self._train_one_attempt(
                    rollout_id=rollout_id,
                    rollout_data_pack=rollout_data_pack,
                    admission_receipt=admission_receipt,
                    attempt=attempt,
                )
            except TrainerCohortChangedError as error:
                cohort_error = error

        try:
            await retry(_fn, max_attempts=_RETRY_MAX_ATTEMPTS)
            if cohort_error is not None:
                raise cohort_error
            self._test_action_executor.run_after_step(rollout_id=rollout_id)
        finally:
            if admission_receipt is not None:
                self.discard_train_batch_admission(admission_receipt)

    async def _train_one_attempt(
        self,
        *,
        rollout_id: int,
        rollout_data_pack,
        admission_receipt: TrainerAdmissionReceipt | None,
        attempt: int,
    ) -> None:
        expected_cells = None
        expected_handles = None
        if admission_receipt is None:
            await self._refresh_cells(rollout_id=rollout_id)
        else:
            expected_cells, expected_handles = self._validate_train_batch_pin(
                rollout_id=rollout_id,
                rollout_data_pack=rollout_data_pack,
                receipt=admission_receipt,
            )

        witness_info = self._allocate_witness_info(
            rollout_id=rollout_id,
            attempt=attempt,
            sample_indices=rollout_data_pack["sample_indices"],
        )
        log_structured(logger.info, op="train", phase="start", rollout=rollout_id, attempt=attempt)
        snapshot_alive_cells, results = await self._execute_all_alive_and_catch(
            "train",
            rollout_id=rollout_id,
            rollout_data_ref=rollout_data_pack["data_ref"],
            witness_info=witness_info,
            attempt=attempt,
            expected_cells=expected_cells,
            expected_actor_handles=expected_handles,
        )
        cohort_errors = [result for result in results if isinstance(result, TrainerCohortChangedError)]
        if cohort_errors:
            raise cohort_errors[0]
        self._check_train_one_attempt(snapshot_alive_cells, results)
        self._log_step_end_event(
            rollout_id=rollout_id,
            snapshot_alive_cells=snapshot_alive_cells,
            results=results,
        )

    async def admit_train_batch(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
    ) -> TrainerAdmissionReceipt:
        """Read-proof one exact publication across a stable v2 cohort.

        A rank reads its own shard when the manager splits the publication by
        data-parallel rank. A rank that has not learned its data-parallel layout reads
        every shard. Under --delay-split-train-data-by-dp the publication is one
        reference, and every rank reads the whole batch. The group checks that the
        receipts together cover every published shard.
        """
        publication = rollout_data_pack.get("trainer_admission")
        if not isinstance(publication, TrainBatchPublication) or publication.rollout_id != rollout_id:
            raise ValueError(f"Invalid trainer admission for rollout {rollout_id}.")
        if "data_ref" not in rollout_data_pack:
            raise ValueError(f"Trainer admission for rollout {rollout_id} has no published data reference.")
        data_ref = rollout_data_pack["data_ref"]
        validate_publication_data_ref(publication, data_ref)

        await self._refresh_cells(rollout_id=rollout_id)
        quorum_id = self._indep_dp_quorum_id
        live_cells = tuple(cell for cell in self._cells if cell.is_alive)
        if not live_cells:
            raise RuntimeError(f"Trainer admission {publication.admission_id} has no live trainer cells.")
        expected_cell_ids = tuple(cell.cell_index for cell in live_cells)
        actor_handle_snapshots = tuple(cell._snapshot_actor_handles() for cell in live_cells)
        expected_ranks = tuple(tuple(range(len(handles))) for handles in actor_handle_snapshots)

        # Admission only proves that every rank can read the exact batch it will
        # consume. That is its own shard when the manager splits the publication by
        # data-parallel rank, and the whole batch when the publication is one reference.
        # A failure here must leave the cohort alive so the source group can be replayed.
        # It is deliberately not a liveness check: a dead actor fails admission loudly
        # and fail-closed, while deciding a cell is gone and retiring it belongs to the
        # heartbeat monitor (cell_monitor.py) and the --use-fault-tolerance controller.
        snapshot_cells, results = await self._execute_all_alive_and_catch(
            "admit_train_batch", publication, data_ref, kill_on_failure=False
        )
        current_live_cells = tuple(cell for cell in self._cells if cell.is_alive)
        response_cells = tuple(snapshot_cells)
        current_cell_ids = tuple(cell.cell_index for cell in current_live_cells)
        response_cell_ids = tuple(cell.cell_index for cell in response_cells)
        cell_identity_stable = _same_identity_sequence(current_live_cells, live_cells) and _same_identity_sequence(
            response_cells, live_cells
        )
        handles_identity_stable = False
        if cell_identity_stable:
            handles_identity_stable = all(
                _same_identity_sequence(cell._snapshot_actor_handles(), expected_handles)
                for cell, expected_handles in zip(live_cells, actor_handle_snapshots, strict=True)
            )
        if (
            self._indep_dp_quorum_id != quorum_id
            or current_cell_ids != expected_cell_ids
            or response_cell_ids != expected_cell_ids
            or len(results) != len(expected_cell_ids)
            or not cell_identity_stable
            or not handles_identity_stable
        ):
            raise RuntimeError(f"Trainer admission {publication.admission_id} received a changed cohort.")

        cohort_cells: list[TrainerCellCohort] = []
        for cell, cell_result, expected in zip(snapshot_cells, results, expected_ranks, strict=True):
            if isinstance(cell_result, BaseException):
                raise RuntimeError(
                    f"Trainer admission {publication.admission_id} failed on a trainer cell."
                ) from cell_result
            if len(cell_result) != len(expected):
                raise RuntimeError(f"Trainer admission {publication.admission_id} missed a trainer rank response.")
            ranks: list[int] = []
            for response in cell_result:
                if not isinstance(response, TrainerRankReceipt) or response.publication != publication:
                    raise RuntimeError(f"Trainer admission {publication.admission_id} received a stale rank response.")
                if response.rank not in expected or response.rank in ranks:
                    raise RuntimeError(
                        f"Trainer admission {publication.admission_id} received a duplicate or foreign rank."
                    )
                ranks.append(response.rank)
            if tuple(sorted(ranks)) != expected:
                raise RuntimeError(f"Trainer admission {publication.admission_id} missed a trainer rank response.")
            # Every cell trains on the whole batch, so each cell's receipts must cover it.
            validate_data_parallel_coverage(cell_result, publication)
            cohort_cells.append(TrainerCellCohort(cell_index=cell.cell_index, ranks=expected))

        receipt = TrainerAdmissionReceipt(
            publication=publication,
            role=self.role,
            cohort=TrainerCohort(quorum_id=quorum_id, cells=tuple(cohort_cells)),
        )
        self._train_batch_pins_for_write()[publication] = (
            receipt,
            quorum_id,
            tuple(live_cells),
            tuple(actor_handle_snapshots),
        )
        return receipt

    def discard_train_batch_admission(self, receipt: TrainerAdmissionReceipt) -> None:
        """Discard the private cohort pin for an admission receipt."""
        pins = self._train_batch_pins_for_write()
        pin = pins.get(receipt.publication)
        if pin is not None and pin[0] == receipt:
            pins.pop(receipt.publication, None)

    def _validate_train_batch_pin(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        receipt: TrainerAdmissionReceipt,
    ) -> tuple[tuple[RayTrainCell, ...], tuple[tuple[object, ...], ...]]:
        publication = rollout_data_pack.get("trainer_admission")
        if publication != receipt.publication or publication.rollout_id != rollout_id:
            raise TrainerCohortChangedError("Trainer admission receipt does not match the train batch publication.")
        try:
            validate_publication_data_ref(publication, rollout_data_pack["data_ref"])
        except (TypeError, ValueError) as error:
            raise TrainerCohortChangedError("Trainer admission data reference changed before training.") from error
        pin = self._train_batch_pins_for_write().get(publication)
        if pin is None or pin[0] != receipt:
            raise TrainerCohortChangedError(f"Trainer admission {publication.admission_id} has no exact trainer pin.")
        expected_quorum_id, expected_cells, expected_handles = pin[1:]
        current_cells = tuple(cell for cell in self._cells if cell.is_alive)
        if self._indep_dp_quorum_id != expected_quorum_id or not _same_identity_sequence(
            current_cells, expected_cells
        ):
            raise TrainerCohortChangedError(
                f"Trainer admission {publication.admission_id} detected trainer cohort drift."
            )
        current_handles = tuple(cell._snapshot_actor_handles() for cell in current_cells)
        if len(current_handles) != len(expected_handles) or any(
            not _same_identity_sequence(current, expected)
            for current, expected in zip(current_handles, expected_handles, strict=True)
        ):
            raise TrainerCohortChangedError(
                f"Trainer admission {publication.admission_id} detected trainer cohort drift."
            )
        return expected_cells, expected_handles

    def _train_batch_pins_for_write(self) -> dict:
        if not hasattr(self, "_train_batch_pins"):
            self._train_batch_pins = {}
        return self._train_batch_pins

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
    def _check_train_one_attempt(snapshot_alive_cells, results):
        outcomes = RayTrainGroup._compute_attempt_outcomes(snapshot_alive_cells, results)
        if not outcomes["normal"] and not outcomes["discarded"]:
            log_structured(logger.error, op="check", **outcomes, decision="retry", reason="all alive cells failed")
            raise RuntimeError("All cells failed in this training attempt")

        # NOTE: If some cells errors + all other cells claim normal, we do *not* retry
        #       This may happen when some cells fails *after* exchanging gradients w/ others
        if outcomes["discarded"]:
            log_structured(logger.warning, op="check", **outcomes, decision="retry", reason="discarded_should_retry")
            raise ValueError("Exists DISCARDED_SHOULD_RETRY, thus need retry")

        log_structured(
            logger.info, op="check", **outcomes, decision="no_retry", reason="survivors normal, gradients valid"
        )

    @staticmethod
    def _compute_attempt_outcomes(snapshot_alive_cells, results) -> dict[str, list[int]]:
        paired = list(zip(snapshot_alive_cells, results, strict=True))
        errored = [c.cell_index for c, r in paired if isinstance(r, BaseException)]
        discarded = [
            c.cell_index
            for c, r in paired
            if not isinstance(r, BaseException) and any(o == TrainStepOutcome.DISCARDED_SHOULD_RETRY for o in r)
        ]
        normal = [c.cell_index for c, r in paired if c.cell_index not in errored and c.cell_index not in discarded]
        return {"errored": errored, "discarded": discarded, "normal": normal}

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
        await retry(
            lambda _: self._execute_first_alive("save_model", rollout_id, force_sync=force_sync),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights to rollout engines."""
        log_structured(logger.info, op="update_weights", phase="start", rollout=rollout_id)
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Fetch the updatable engines + lock once (like V1 RayActorGroup) so all
        # ranks observe a consistent engine set; the actor releases the lock itself.
        info = await self._rollout_manager.get_updatable_engines_and_lock.remote()
        await self._rollout_manager.health_monitoring_pause.remote()
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        await retry(
            lambda _: self._execute_first_alive("update_weights", info=info),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

        await self._maybe_log_inference_engine_weight_checksums(rollout_id=rollout_id)

    async def _maybe_log_inference_engine_weight_checksums(self, *, rollout_id: int | None) -> None:
        if not is_event_logger_initialized():
            return
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        check_weights_result = await self._rollout_manager.check_weights.remote("checksum")
        engine_checksums = flatten_inference_engine_checksums(check_weights_result)
        get_event_logger().log(
            InferenceEngineWeightChecksumEvent,
            dict(rollout_id=rollout_id, engine_checksums=engine_checksums),
        )

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

    async def set_rollout_manager(self):
        await asyncio.gather(*[cell.set_rollout_manager() for cell in self._cells])

    def stop_cell(self, cell_index: int) -> None:
        self._cells[cell_index].stop()

    def start_cell(self, cell_index: int) -> None:
        """Mark a stopped cell as pending. Actual startup happens in train()."""
        self._cells[cell_index].mark_as_pending()

    # ------------------------ utils to forward calls to cells ------------------------

    async def _execute_all_alive_and_catch(
        self,
        fn_name: str,
        *args,
        expected_cells: tuple[RayTrainCell, ...] | None = None,
        expected_actor_handles: tuple[tuple[object, ...], ...] | None = None,
        **kwargs,
    ):
        snapshot_alive_cells = [c for c in self._cells if c.is_alive]
        assert snapshot_alive_cells, "No alive cells"
        if expected_cells is not None and not _same_identity_sequence(snapshot_alive_cells, expected_cells):
            raise TrainerCohortChangedError("Trainer cells changed after admission and before training dispatch.")
        if expected_actor_handles is not None and len(expected_actor_handles) != len(snapshot_alive_cells):
            raise TrainerCohortChangedError(
                "Trainer actor handles changed after admission and before training dispatch."
            )
        # NOTE: no timeout here. If a cell hangs, the external FT controller
        # detects stale heartbeat via cell_status(), calls cell.stop() to kill
        # actors, which unblocks this gather with ActorDiedError.
        outputs = await asyncio.gather(
            *[
                cell.execute(
                    fn_name,
                    *args,
                    expected_actor_handles=(
                        None if expected_actor_handles is None else expected_actor_handles[cell_index]
                    ),
                    **kwargs,
                )
                for cell_index, cell in enumerate(snapshot_alive_cells)
            ],
            return_exceptions=True,
        )
        AsyncioGatherUtils.log_error(outputs, debug_name=f"execute_all_alive_and_catch#{fn_name}")
        return snapshot_alive_cells, outputs

    async def _execute_first_alive(self, fn_name: str, *args, **kwargs):
        alive_cells = [c for c in self._cells if c.is_alive]
        assert alive_cells, "No alive cells, therefore cannot heal anymore"
        return await alive_cells[0].execute(fn_name, *args, **kwargs)

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self, *, rollout_id: int) -> None:
        snapshotted_pending_indices = [c.cell_index for c in self._cells if c.is_pending]
        snapshotted_alive_indices = [c.cell_index for c in self._cells if c.is_alive]
        will_alive_indices = sorted(list(set(snapshotted_pending_indices + snapshotted_alive_indices)))
        all_states = [(c.cell_index, c.state_name) for c in self._cells]
        log_structured(
            logger.info,
            op="refresh",
            phase="start",
            rollout=rollout_id,
            alive=snapshotted_alive_indices,
            pending=snapshotted_pending_indices,
            all_states=all_states,
            quorum=self._indep_dp_quorum_id,
        )
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
            log_structured(
                logger.info,
                op="refresh",
                phase="decision",
                rollout=rollout_id,
                needs_reconfigure=False,
                reason="alive_config_unchanged,no_pending",
                quorum=self._indep_dp_quorum_id,
            )
            return
        reason = "+".join(
            r
            for r, on in [
                ("pending_cell", exists_pending_cell),
                ("alive_config_changed", exists_alive_cell_changed_config),
            ]
            if on
        )
        log_structured(
            logger.info,
            op="refresh",
            phase="decision",
            rollout=rollout_id,
            needs_reconfigure=True,
            reason=reason,
            will_alive=will_alive_indices,
            quorum_from=self._indep_dp_quorum_id,
            quorum_to=self._indep_dp_quorum_id + 1,
        )

        # Step 1: Bump states
        self._indep_dp_quorum_id += 1

        # Step 2: Allocate pending actors
        # We currently do not consider this phase to have errors (because it does not touch GPUs)
        for c in self._cells:
            if c.cell_index in snapshotted_pending_indices:
                c.allocate_for_pending()

        # Step 3: Cooperatively prepare
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
            log_structured(
                logger.info,
                op="refresh",
                phase="end",
                rollout=rollout_id,
                quorum=self._indep_dp_quorum_id,
                alive=will_alive_indices,
                healed=snapshotted_pending_indices,
                reconfigured=True,
            )
            self._log_reconfigure_event(
                rollout_id=rollout_id,
                src_cell_index=src_cell_index if snapshotted_pending_indices else None,
                healed_cell_indices=snapshotted_pending_indices,
                alive_cell_indices_after=will_alive_indices,
            )
        else:
            log_structured(
                logger.error,
                op="refresh",
                phase="end",
                rollout=rollout_id,
                reconfigured=False,
                quorum=self._indep_dp_quorum_id,
                reason="cooperative_prepare_raised",
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
