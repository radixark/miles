import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.specs.train import compute_trainer_num_cells, compute_trainer_pool_id
from miles.ray.train.cell import RayTrainCell
from miles.ray.train.cell_monitor import create_trainer_cell_health_checker
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
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import ActivenessTracker, NoopHealthChecker, SimpleHealthCheckerConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import NonRetryableError, retry, retry_until_deadline
from miles.utils.test_utils.ft_test_actions import FTTestActionControllerExecutor
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)


_RETRY_MAX_ATTEMPTS = 30
_CELLS_READY_TIMEOUT_SECONDS = 3600.0


class RayTrainGroup:
    def __init__(
        self,
        args,
        *,
        inference_controller: object | None,
        rollout_executor: object | None,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._inference_controller = inference_controller
        self._rollout_executor = rollout_executor
        self._role = role
        self._with_ref = with_ref
        self._with_opd_teacher = with_opd_teacher
        self._pool_id = compute_trainer_pool_id(role)
        self._watcher_disposer: StopWatchFn | None = None

        self._indep_dp_quorum_id = 0

        self._health_checker_config = (
            SimpleHealthCheckerConfig.from_args(args, prefix="trainer_heartbeat_checker")
            if self._expected_num_cells > 1
            else None
        )

        self._health_checker_activeness = ActivenessTracker(active=True)

        self._cells_by_id: dict[str, RayTrainCell] = {}

        self._witness_allocator: WitnessIdAllocator | None = (
            WitnessIdAllocator(buffer_size=args.witness_buffer_size) if args.enable_witness else None
        )
        if self._witness_allocator is not None and args.save_debug_event_data is not None:
            self._witness_allocator.resume(read_persisted_witness_counter(Path(args.save_debug_event_data)))

        self._test_action_executor = FTTestActionControllerExecutor.from_args(args, group=self)

    @property
    def pool_id(self) -> str:
        return self._pool_id

    @property
    def expected_num_cells(self) -> int:
        return self._expected_num_cells

    @property
    def _expected_num_cells(self) -> int:
        return compute_trainer_num_cells(self.args, role=self._role)

    @property
    def _cells(self) -> list[RayTrainCell]:
        return sorted(self._cells_by_id.values(), key=lambda cell: cell.cell_index)

    @property
    def cell_ids(self) -> list[str]:
        return [cell.cell_id for cell in self._cells]

    async def _wait_expected_num_cells(self, timeout: float = _CELLS_READY_TIMEOUT_SECONDS) -> None:
        async def _check(_remaining: float) -> None:
            expected = self._expected_num_cells
            if len(self._cells_by_id) < expected:
                raise TimeoutError(f"only {len(self._cells_by_id)} of {expected} trainer cells observed")

        await retry_until_deadline(
            _check,
            total_seconds=timeout,
            retry_on=TimeoutError,
            initial_delay=1.0,
            max_delay=5.0,
            log_fields=dict(tag="ft", spec=self._pool_id),
        )

    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        actual = self._cells_by_id.get(cell_id)
        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual.workers_hash if actual is not None else None,
            add=self._add_cell,
            remove=self._remove_cell,
        )

    async def _add_cell(self, cell_id: str, observed: CellInfo) -> None:
        self._cells_by_id[cell_id] = self._create_cell(
            cell_id, cell_index=observed.meta["cell_index"], workers_hash=observed.workers_hash
        )

    async def _remove_cell(self, cell_id: str) -> None:
        cell = self._cells_by_id.pop(cell_id)
        cell.health_checker.stop()

    def _create_cell(self, cell_id: str, *, cell_index: int, workers_hash: str) -> RayTrainCell:
        cell = RayTrainCell(
            args=self.args,
            role=self._role,
            with_ref=self._with_ref,
            with_opd_teacher=self._with_opd_teacher,
            cell_id=cell_id,
            cell_index=cell_index,
            workers_hash=workers_hash,
            rollout_executor=self._rollout_executor,
            health_checker=NoopHealthChecker(),
        )

        if self._health_checker_config is not None:
            cell.health_checker = create_trainer_cell_health_checker(
                cell=cell,
                config=self._health_checker_config,
                get_activeness=self._health_checker_activeness.get,
            )

        return cell

    async def dispose(self) -> None:
        if (disposer := self._watcher_disposer) is not None:
            await disposer()
            self._watcher_disposer = None

    # ------------------------ API :: train ------------------------

    async def train(self, rollout_id: int, rollout_data_pack, external_data: list | None = None) -> list:
        """Do one rollout training"""

        assert (
            external_data is None or len(self._cells) == 1
        ), "external_data is only supported for a single cell, i.e. without independent DP"

        event_analyzer.run_analysis_from_args(self.args)

        async def _fn(attempt: int) -> list:
            witness_info = self._allocate_witness_info(
                rollout_id=rollout_id,
                attempt=attempt,
                sample_indices=rollout_data_pack["sample_indices"],
            )

            log_structured(logger.info, tag="ft", op="train", phase="start", rollout=rollout_id, attempt=attempt)
            await self._refresh_cells(rollout_id=rollout_id)
            snapshot_alive_cells, results = await self._gather_all_alive_and_catch(
                lambda cell: cell.train(
                    rollout_id=rollout_id,
                    rollout_data_ref=rollout_data_pack["data_ref"],
                    witness_info=witness_info,
                    attempt=attempt,
                    external_data=external_data,
                ),
                debug_name="execute_all_alive_and_catch#train",
                check_recoverable=False,
            )
            self._check_train_one_attempt(snapshot_alive_cells, results)

            self._log_step_end_event(
                rollout_id=rollout_id,
                snapshot_alive_cells=snapshot_alive_cells,
                results=results,
            )

            return [
                worker_result
                for cell_results in results
                if not isinstance(cell_results, BaseException)
                for worker_result in cell_results
            ]

        worker_results = await retry(_fn, max_attempts=_RETRY_MAX_ATTEMPTS)

        await self._test_action_executor.run_after_step(rollout_id=rollout_id)

        return worker_results

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
                cell.cell_index: (
                    "error" if isinstance(cell_results, BaseException) else [r.outcome for r in cell_results]
                )
                for cell, cell_results in zip(snapshot_alive_cells, results, strict=True)
            }
            get_event_logger().log(
                TrainGroupStepEndEvent,
                dict(rollout_id=rollout_id, cell_outcomes=cell_outcomes),
            )

    def _check_train_one_attempt(self, snapshot_alive_cells, results):
        outcomes = RayTrainGroup._compute_attempt_outcomes(snapshot_alive_cells, results)
        if not outcomes["normal"] and not outcomes["discarded"]:
            cause = _first_exception(results)
            error = self._make_all_cells_failed_error("All cells failed in this training attempt", cause=cause)
            log_structured(
                logger.error,
                tag="ft",
                op="check",
                **outcomes,
                decision="give_up" if isinstance(error, NonRetryableError) else "retry",
                reason="all alive cells failed",
            )
            raise error from cause

        # NOTE: If some cells errors + all other cells claim normal, we do *not* retry
        #       This may happen when some cells fails *after* exchanging gradients w/ others
        if outcomes["discarded"]:
            log_structured(
                logger.warning,
                tag="ft",
                op="check",
                **outcomes,
                decision="retry" if self._is_recoverable() else "give_up",
                reason="discarded_should_retry",
            )
            raise ValueError("Exists DISCARDED_SHOULD_RETRY, thus need retry")

        log_structured(
            logger.info,
            tag="ft",
            op="check",
            **outcomes,
            decision="no_retry",
            reason="survivors normal, gradients valid",
        )

    @staticmethod
    def _compute_attempt_outcomes(snapshot_alive_cells, results) -> dict[str, list[int]]:
        paired = list(zip(snapshot_alive_cells, results, strict=True))
        errored = [c.cell_index for c, r in paired if isinstance(r, BaseException)]
        discarded = [
            c.cell_index
            for c, r in paired
            if not isinstance(r, BaseException)
            and any(o.outcome == TrainStepOutcome.DISCARDED_SHOULD_RETRY for o in r)
        ]
        normal = [c.cell_index for c, r in paired if c.cell_index not in errored and c.cell_index not in discarded]
        return {"errored": errored, "discarded": discarded, "normal": normal}

    # ------------------------ API :: others ------------------------

    async def init(self):
        """
        Observe the controller's cells, then allocate GPU resources and initialize
        model, optimzier, local ckpt, etc.
        """
        provider: BaseWorkerProvider = RayWorkerProvider.create(pool_ids=[self._pool_id])
        self._watcher_disposer = await provider.watch_cells(self._reconcile)
        await self._wait_expected_num_cells()

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
            lambda _: self._execute_first_alive("save_model", rollout_id=rollout_id, force_sync=force_sync),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

    async def export_hf(self, rollout_id: int, path: str):
        """Export current weights as an HF checkpoint. Only cell 0 exports to avoid file write conflicts."""
        await retry(
            lambda _: self._execute_first_alive("export_hf", rollout_id=rollout_id, path=path),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )

    async def update_weights(self, rollout_id: int | None = None):
        """Broadcast weights to rollout engines."""
        log_structured(logger.info, tag="ft", op="update_weights", phase="start", rollout=rollout_id)
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Fetch the updatable engines once (like V1 RayActorGroup) so all
        # ranks observe a consistent engine set.
        info = await self._inference_controller.start_update_weights()
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        await retry(
            lambda _: self._execute_first_alive("update_weights", info=info),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )
        await self._inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        await self._maybe_log_inference_engine_weight_checksums(rollout_id=rollout_id)

    async def _maybe_log_inference_engine_weight_checksums(self, *, rollout_id: int | None) -> None:
        if not is_event_logger_initialized():
            return
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        check_weights_result = await self._inference_controller.check_weights("checksum")
        engine_checksums = flatten_inference_engine_checksums(check_weights_result)
        get_event_logger().log(
            InferenceEngineWeightChecksumEvent,
            dict(rollout_id=rollout_id, engine_checksums=engine_checksums),
        )

    async def onload(self):
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("wake_up")
        self._health_checker_activeness.bump_active(True)

    @contextmanager
    def _paused_health_checkers(self) -> Iterator[None]:
        self._health_checker_activeness.bump_active(False)
        try:
            yield
        finally:
            self._health_checker_activeness.bump_active(True)

    async def offload(self):
        self._health_checker_activeness.bump_active(False)
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("sleep")

    async def clear_memory(self):
        # Catch *without* retry: cells w/ exceptions are auto marked errored, and will not be used
        await self._execute_all_alive_and_catch("clear_memory")

    async def reconcile_adapters(self) -> None:
        await asyncio.gather(*[cell.execute("reconcile_adapters") for cell in self._cells])

    async def set_rollout_executor(self):
        await asyncio.gather(*[cell.set_rollout_executor() for cell in self._cells])

    def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {cell_id: cell.cell_status() for cell_id, cell in list(self._cells_by_id.items())}

    # ------------------------ utils to forward calls to cells ------------------------

    def _is_recoverable(self) -> bool:
        return any(cell.is_alive for cell in self._cells)

    def _make_all_cells_failed_error(self, message: str, *, cause: BaseException | None = None) -> Exception:
        if isinstance(cause, NonRetryableError):
            return NonRetryableError(message)
        return RuntimeError(message) if self._is_recoverable() else NonRetryableError(message)

    def _raise_if_no_cell_can_recover(self, *, debug_name: str, cause: BaseException | None) -> None:
        if self._is_recoverable():
            return
        raise self._make_all_cells_failed_error(f"All cells failed during {debug_name}", cause=cause) from cause

    async def _execute_all_alive_and_catch(self, fn_name: str, *, check_recoverable: bool = True, **kwargs):
        return await self._gather_all_alive_and_catch(
            lambda cell: cell.execute(fn_name, **kwargs),
            debug_name=f"execute_all_alive_and_catch#{fn_name}",
            check_recoverable=check_recoverable,
        )

    async def _gather_all_alive_and_catch(self, compute_coroutine, *, debug_name: str, check_recoverable: bool = True):
        snapshot_alive_cells = [c for c in self._cells if c.is_alive]
        if not snapshot_alive_cells:
            raise NonRetryableError("No alive cells")
        # NOTE: no timeout here. If a cell hangs, the external FT controller
        # detects stale heartbeat via cell_status() and suspends the cell through
        # the worker manager, which unblocks this gather with ActorDiedError.
        outputs = await asyncio.gather(
            *[compute_coroutine(cell) for cell in snapshot_alive_cells],
            return_exceptions=True,
        )
        AsyncioGatherUtils.log_error(outputs, debug_name=debug_name)
        if check_recoverable:
            self._raise_if_no_cell_can_recover(debug_name=debug_name, cause=_first_exception(outputs))
        return snapshot_alive_cells, outputs

    async def _execute_first_alive(self, fn_name: str, **kwargs):
        alive_cells = [c for c in self._cells if c.is_alive]
        if not alive_cells:
            raise NonRetryableError("No alive cells, therefore cannot heal anymore")
        try:
            return await alive_cells[0].execute(fn_name, **kwargs)
        except Exception as cause:
            self._raise_if_no_cell_can_recover(debug_name=f"execute_first_alive#{fn_name}", cause=cause)
            raise

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self, *, rollout_id: int) -> None:
        snapshotted_healing_indices = [c.cell_index for c in self._cells if c.is_uninitialized]
        snapshotted_alive_indices = [c.cell_index for c in self._cells if c.is_alive]
        will_alive_indices = sorted(list(set(snapshotted_healing_indices + snapshotted_alive_indices)))
        all_states = [(c.cell_index, c.state_name) for c in self._cells]
        log_structured(
            logger.info,
            tag="ft",
            op="refresh",
            phase="start",
            rollout=rollout_id,
            alive=snapshotted_alive_indices,
            healing=snapshotted_healing_indices,
            all_states=all_states,
            quorum=self._indep_dp_quorum_id,
        )
        if not snapshotted_alive_indices:
            raise NonRetryableError("Cannot recover when all cells are dead")

        # Step 0: Determine whether need to reconfigure
        exists_alive_cell_changed_config = any(
            cell.indep_dp_info.alive_cell_indices != will_alive_indices
            for cell in self._cells
            if cell.cell_index in snapshotted_alive_indices
        )
        exists_healing_cell = len(snapshotted_healing_indices) != 0
        needs_reconfigure = exists_healing_cell or exists_alive_cell_changed_config
        if not needs_reconfigure:
            log_structured(
                logger.info,
                tag="ft",
                op="refresh",
                phase="decision",
                rollout=rollout_id,
                needs_reconfigure=False,
                reason="alive_config_unchanged,no_healing",
                quorum=self._indep_dp_quorum_id,
            )
            return
        reason = "+".join(
            r
            for r, on in [
                ("healing_cell", exists_healing_cell),
                ("alive_config_changed", exists_alive_cell_changed_config),
            ]
            if on
        )
        log_structured(
            logger.info,
            tag="ft",
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

        # Step 2: Cooperatively prepare
        src_cell_index = snapshotted_alive_indices[0]  # TODO make it balanced, and support multi-src-to-one-dst
        src_alive_rank = will_alive_indices.index(src_cell_index)
        ckpt_dst_alive_ranks = [will_alive_indices.index(x) for x in snapshotted_healing_indices]

        with self._paused_health_checkers():
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
                            recv_ckpt_src_rank=src_alive_rank if c.cell_index in snapshotted_healing_indices else None,
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
                tag="ft",
                op="refresh",
                phase="end",
                rollout=rollout_id,
                quorum=self._indep_dp_quorum_id,
                alive=will_alive_indices,
                healed=snapshotted_healing_indices,
                reconfigured=True,
            )
            self._log_reconfigure_event(
                rollout_id=rollout_id,
                src_cell_index=src_cell_index if snapshotted_healing_indices else None,
                healed_cell_indices=snapshotted_healing_indices,
                alive_cell_indices_after=will_alive_indices,
            )
        else:
            log_structured(
                logger.error,
                tag="ft",
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


def _first_exception(results) -> BaseException | None:
    return next((result for result in results if isinstance(result, BaseException)), None)
