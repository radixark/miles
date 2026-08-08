"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TypeVar, cast

from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    TrainBatchLease,
    TrainBatchRollbackReason,
)
from miles.rollout.data_source import DataSource, SourceReservation
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DataBufferSource,
    DefaultDataBuffer,
    Group,
    first_sample,
)
from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncRetryReason,
)
from miles.rollout.fully_async.ownership import ReservationOwnership, ReservationStageId, ReservationTerminalReceipt
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.fully_async import InferenceFullyAsyncExecutor
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0
_T = TypeVar("_T")


@dataclass(frozen=True)
class _OwnedCompletedGroup:
    terminal_receipt: ReservationTerminalReceipt
    samples: Group
    expected_parent_identities: tuple[tuple[int | None, int | None], ...]


@dataclass(frozen=True)
class _OwnedExecutionFailure:
    terminal_receipt: ReservationTerminalReceipt
    error: BaseException


@dataclass(frozen=True)
class _OwnedExecutionRetry:
    terminal_receipt: ReservationTerminalReceipt
    reason: FullyAsyncRetryReason


_OwnedTerminalResult = _OwnedCompletedGroup | _OwnedExecutionRetry | _OwnedExecutionFailure
_OwnedTerminalObserver = Callable[[], Coroutine[object, object, _OwnedTerminalResult]]


@dataclass(frozen=True)
class _ActiveOwnedExecution:
    execution: FullyAsyncExecution
    observe_terminal: _OwnedTerminalObserver


class _OwnedTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        *,
        rollout_id: int,
        owner: "FullyAsyncRolloutFn",
        terminal_receipts: list[ReservationTerminalReceipt],
    ) -> None:
        super().__init__(rollout_id=rollout_id)
        self._owner_loop = asyncio.get_running_loop()
        self._owner = owner
        self._terminal_receipts = terminal_receipts

    def _commit(self) -> None:
        self._run_on_owner_loop(self._commit_on_owner_loop)

    def _commit_on_owner_loop(self) -> None:
        self._owner._commit_owned_terminals(self._terminal_receipts, rollout_id=self.rollout_id)

    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        self._run_on_owner_loop(lambda: self._owner._rollback_owned_terminals(self._terminal_receipts))

    def _run_on_owner_loop(self, operation: Callable[[], None]) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._owner_loop:
            operation()
            return

        completion: Future[None] = Future()

        def run_operation() -> None:
            try:
                operation()
            except BaseException as error:
                completion.set_exception(error)
            else:
                completion.set_result(None)

        self._owner_loop.call_soon_threadsafe(run_operation)
        completion.result()


def _supports_source_reservations(data_source: DataSource) -> bool:
    """Whether ``data_source`` declares durable source reservations.

    Overriding ``reserve_samples`` is not a capability signal: sources such as
    ``RolloutDataSourceWithBuffer`` override it only to reject reservations, so
    a source has to opt in explicitly.
    """
    return bool(getattr(data_source, "supports_source_reservations", False))


def _owned_group_identity_error(completed: _OwnedCompletedGroup) -> ValueError | None:
    reservation_id = completed.terminal_receipt.executor_receipt.reservation_id
    if len(completed.samples) != len(completed.expected_parent_identities):
        return ValueError(f"Source reservation {reservation_id} returned {len(completed.samples)} parent slots; expected {len(completed.expected_parent_identities)}.")
    for position, expected_identity in enumerate(completed.expected_parent_identities):
        item = completed.samples[position]
        samples = item if isinstance(item, list) else [item]
        if any(not isinstance(sample, Sample) for sample in samples):
            return ValueError(f"Source reservation {reservation_id} returned non-Sample values at parent slot {position}.")
        actual_identities = [(sample.group_index, sample.index) for sample in samples]
        if not actual_identities or any(identity != expected_identity for identity in actual_identities):
            return ValueError(f"Source reservation {reservation_id} returned sample identities {actual_identities} at parent slot {position}; expected every sample to have identity {expected_identity}.")
    return None


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._uses_owned_scheduling = _supports_source_reservations(self.data_source)
        if not self._uses_owned_scheduling:
            logger.warning(
                "Data source %s does not support source reservations; "
                "fully async rollout schedules work without source ownership.",
                type(self.data_source).__name__,
            )
        execution_samples = self.args.async_max_concurrent_samples
        if execution_samples is None:
            execution_samples = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        self._max_execution_groups = execution_samples // self.args.n_samples_per_prompt
        buffer_groups = int(self.args.async_data_buffer_capacity_factor * self.args.rollout_batch_size)
        self._max_completed_groups = max(buffer_groups, self.args.rollout_batch_size)
        self._retained_slots = asyncio.BoundedSemaphore(self._max_execution_groups + self._max_completed_groups) if self._uses_owned_scheduling else None
        self._completed_slots: asyncio.Queue[object] | None = None
        self._completed_slot_available = asyncio.Event()
        self._owned_capacity_released = asyncio.Event()
        self._ownership = ReservationOwnership(self.data_source) if self._uses_owned_scheduling else None
        self._executor = (
            InferenceFullyAsyncExecutor(
                self.state,
                sample_done_callback=self._scheduler.sample_done_callback,
            )
            if self._uses_owned_scheduling
            else None
        )
        self._active_executions: dict[asyncio.Task[_OwnedTerminalResult], _ActiveOwnedExecution] = {}
        self._pending_reserved_rollbacks: list[SourceReservation] = []
        self._pending_terminal_rollbacks: list[tuple[ReservationTerminalReceipt, bool]] = []
        self._discarded_terminal_receipts: deque[ReservationTerminalReceipt] = deque()
        self._discarded_terminal_available = asyncio.Event()
        self._next_execution_id = 1
        if self._uses_owned_scheduling:
            self._completed_slots = asyncio.Queue(maxsize=self._max_completed_groups)
            for _ in range(self._max_completed_groups):
                self._completed_slots.put_nowait(object())
            self._completed_slot_available.set()

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or DefaultDataBuffer
            self._output = buffer_cls(
                DataBufferConstructorInput(
                    args=self.args,
                    unused_handler_fn=self._handle_unused_buffer_source,
                    discard_handler_fn=self._discard_buffer_source,
                )
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if self._uses_owned_scheduling:
            return self._max_execution_groups
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task[DataBufferInput | _OwnedTerminalResult]:
        if not self._uses_owned_scheduling:
            samples = self.data_source.get_samples(1)
            self._scheduler.on_submit(samples)
            [prompt_group] = samples
            return asyncio.create_task(self._generate_group(prompt_group))

        ownership = self._require_ownership()
        retained_slots = self._require_retained_slots()
        try:
            [reservation] = ownership.reserve_samples(1)
        except Exception:
            if not ownership.has_pending_acquisition_rollback:
                retained_slots.release()
            raise

        try:
            expected_parent_identities = self._validate_reservation(reservation)
            stage_id = ReservationStageId(f"execution-{self._next_execution_id}")
            self._next_execution_id += 1
            [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
        except Exception as validation_error:
            try:
                ownership.rollback_reserved([reservation])
            except BaseException as rollback_error:
                self._pending_reserved_rollbacks.append(reservation)
                raise validation_error from rollback_error
            retained_slots.release()
            raise

        executor = self._executor
        if executor is None:
            raise RuntimeError("Fully async executor is not initialized.")
        try:
            execution = executor.submit(reservation, executor_receipt)
        except BaseException as submission_error:
            try:
                [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
                ownership.rollback_batch([terminal_receipt])
            except BaseException as settlement_error:
                if "terminal_receipt" in locals():
                    self._pending_terminal_rollbacks.append((terminal_receipt, False))
                raise submission_error from settlement_error
            retained_slots.release()
            raise
        self._scheduler.on_submit([list(reservation.samples)])

        async def observe_terminal() -> _OwnedTerminalResult:
            outcome = await execution.wait_terminal()
            if outcome.executor_receipt is not executor_receipt:
                raise RuntimeError(f"Execution receipt {executor_receipt.receipt_id} did not return its exact terminal receipt.")
            [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
            if isinstance(outcome, FullyAsyncExecutionFailure):
                return _OwnedExecutionFailure(terminal_receipt=terminal_receipt, error=outcome.error)
            if isinstance(outcome, FullyAsyncExecutionRetry):
                return _OwnedExecutionRetry(terminal_receipt=terminal_receipt, reason=outcome.reason)
            if not isinstance(outcome, FullyAsyncExecutionSuccess):
                raise RuntimeError(f"Fully async execution returned unsupported {type(outcome).__name__}.")
            completed = _OwnedCompletedGroup(
                terminal_receipt=terminal_receipt,
                samples=cast(Group, outcome.samples),
                expected_parent_identities=expected_parent_identities,
            )
            identity_error = _owned_group_identity_error(completed)
            if identity_error is not None:
                return _OwnedExecutionFailure(terminal_receipt=terminal_receipt, error=identity_error)
            return completed

        terminal_task = asyncio.create_task(observe_terminal())
        self._active_executions[terminal_task] = _ActiveOwnedExecution(
            execution=execution,
            observe_terminal=observe_terminal,
        )
        return terminal_task

    def _validate_reservation(
        self,
        reservation: SourceReservation,
    ) -> tuple[tuple[int | None, int | None], ...]:
        expected_parents = self.args.n_samples_per_prompt
        if len(reservation.samples) != expected_parents:
            raise ValueError(f"Source reservation {reservation.reservation_id} contains {len(reservation.samples)} parent slots; expected {expected_parents}.")
        identities = tuple((sample.group_index, sample.index) for sample in reservation.samples)
        if any(group_index is None or sample_index is None for group_index, sample_index in identities):
            raise ValueError(f"Source reservation {reservation.reservation_id} has incomplete parent identities.")
        if len(identities) != len(set(identities)):
            raise ValueError(f"Source reservation {reservation.reservation_id} has duplicate parent identities: {list(identities)}.")
        return identities

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(source=prompt_group, group=result)

    async def _acquire_retained_slot(self) -> bool:
        retained_slots = self._retained_slots
        if retained_slots is None:
            return False
        await retained_slots.acquire()
        if self._producer_resumed.is_set():
            return True
        retained_slots.release()
        return False

    async def _submit_active_group(
        self,
        active: set[asyncio.Task[DataBufferInput | _OwnedTerminalResult]],
    ) -> bool:
        if self._uses_owned_scheduling and not await self._acquire_retained_slot():
            return False
        active.add(self._submit_one_group())
        return True

    def _completed_capacity_available(self) -> bool:
        return self._completed_slots is None or not self._completed_slots.empty()

    def _retained_capacity_available(self) -> bool:
        return self._retained_slots is None or not self._retained_slots.locked()

    def _try_acquire_completed_slot(self) -> bool:
        completed_slots = self._completed_slots
        if completed_slots is None:
            return False
        try:
            completed_slots.get_nowait()
        except asyncio.QueueEmpty:
            self._completed_slot_available.clear()
            return False
        if completed_slots.empty():
            self._completed_slot_available.clear()
        return True

    async def _wait_for_worker_progress(
        self,
        active: set[asyncio.Task[DataBufferInput | _OwnedTerminalResult]],
        *,
        capacity_blocked: bool,
        scheduler_blocked: bool,
    ) -> tuple[
        set[asyncio.Task[DataBufferInput | _OwnedTerminalResult]],
        set[asyncio.Task[DataBufferInput | _OwnedTerminalResult]],
    ]:
        if capacity_blocked:
            capacity_waiter = asyncio.create_task(self._owned_capacity_released.wait())
            try:
                ready, _ = await asyncio.wait([*active, capacity_waiter], return_when=asyncio.FIRST_COMPLETED)
            finally:
                capacity_waiter.cancel()
                await asyncio.gather(capacity_waiter, return_exceptions=True)
            done = active.intersection(ready)
            return done, active.difference(done)
        if scheduler_blocked:
            return await self._scheduler.wait_for_progress(active)
        return await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)

    async def _worker_loop(self) -> None:
        output = self._output
        if output is None:
            raise RuntimeError("Fully async output buffer is not initialized.")
        active: set[asyncio.Task[DataBufferInput | _OwnedTerminalResult]] = set()
        fatal_error: BaseException | None = None
        while True:
            capacity_blocked = False
            scheduler_blocked = False
            self._owned_capacity_released.clear()
            await self._producer_resumed.wait()
            while fatal_error is None:
                if self._uses_owned_scheduling and not self._completed_capacity_available():
                    capacity_blocked = True
                    break
                if self._uses_owned_scheduling and active and not self._retained_capacity_available():
                    capacity_blocked = True
                    break
                if not self._scheduler.has_capacity(
                    pending_groups=len(active),
                    group_budget=self._max_in_flight_groups(),
                ):
                    scheduler_blocked = True
                    break
                try:
                    submitted = await self._submit_active_group(active)
                except BaseException as error:
                    fatal_error = error
                    break
                if not submitted:
                    break
                if self._uses_owned_scheduling:
                    await asyncio.sleep(0)
                    if any(task.done() for task in active):
                        break
            if not active:
                if fatal_error is not None:
                    raise fatal_error
                if self._uses_owned_scheduling and not self._completed_capacity_available():
                    await self._completed_slot_available.wait()
                    continue
                raise RuntimeError("Fully async scheduler has admission capacity but no active work.")
            done, active = await self._wait_for_worker_progress(
                active,
                capacity_blocked=capacity_blocked,
                scheduler_blocked=scheduler_blocked,
            )
            for task in done:
                if not self._uses_owned_scheduling:
                    await output.put(cast(DataBufferInput, task.result()))
                    continue
                owned_task = cast(asyncio.Task[_OwnedTerminalResult], task)
                try:
                    result = owned_task.result()
                except BaseException as error:
                    fatal_error = fatal_error or error
                    continue
                if isinstance(result, (_OwnedExecutionRetry, _OwnedExecutionFailure)):
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
                    except BaseException as error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, False))
                        fatal_error = fatal_error or error
                    self._active_executions.pop(owned_task, None)
                    if isinstance(result, _OwnedExecutionFailure):
                        fatal_error = fatal_error or result.error
                    continue
                if not isinstance(result, _OwnedCompletedGroup):
                    fatal_error = fatal_error or RuntimeError(f"Fully async execution returned unsupported {type(result).__name__}.")
                    continue
                if fatal_error is not None or not self._try_acquire_completed_slot():
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=False)
                    except BaseException as error:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, False))
                        fatal_error = fatal_error or error
                    self._active_executions.pop(owned_task, None)
                    continue
                try:
                    await output.put(DataBufferInput(source=result.terminal_receipt, group=result.samples))
                except BaseException as error:
                    try:
                        self._rollback_owned_terminal(result.terminal_receipt, completed_slot_held=True)
                    except BaseException:
                        self._pending_terminal_rollbacks.append((result.terminal_receipt, True))
                    fatal_error = fatal_error or error
                finally:
                    self._active_executions.pop(owned_task, None)

    # -------------------------- consumer --------------------------

    async def _next_group(self, input: RolloutFnTrainInput) -> DataBufferInput:
        output = self._output
        worker = self._worker
        if output is None or worker is None:
            raise RuntimeError("Fully async worker is not initialized.")
        queue_get = asyncio.create_task(output.get(current_version=input.weight_version))
        discarded_wait = asyncio.create_task(self._discarded_terminal_available.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, worker, discarded_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                if worker in done:
                    worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if discarded_wait in done:
                    self._flush_discarded_terminals(input.rollout_id)
                    discarded_wait = asyncio.create_task(self._discarded_terminal_available.wait())
                    continue
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            for task in (queue_get, discarded_wait):
                if not task.done():
                    task.cancel()
            await asyncio.gather(queue_get, discarded_wait, return_exceptions=True)

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        terminal_receipts: list[ReservationTerminalReceipt] = []
        do_print = True
        try:
            while len(data) < target_data_size:
                entry = await self._next_group(input)
                if len(entry.group) != args.n_samples_per_prompt:
                    raise ValueError(f"Generated group contains {len(entry.group)} parent slots; expected {args.n_samples_per_prompt}.")
                if isinstance(entry.source, ReservationTerminalReceipt):
                    terminal_receipts.append(entry.source)

                if do_print:
                    sample = first_sample(entry.group)
                    logger.info(
                        "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                        sample_text_preview(sample),
                        str(sample.label)[:100],
                        reward_log_summary(sample.reward),
                    )
                    do_print = False
                data.append(entry.group)

            sample = first_sample(data[-1])
            logger.info(
                "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
                sample_text_preview(sample),
                str(sample.label)[:100],
                reward_log_summary(sample.reward),
            )
            data.sort(key=lambda group: first_sample(group).index)
            if self._sample_filter is not None:
                self._sample_filter(args, data)

            metrics = self._output.get_metrics()
            if terminal_receipts:
                return LeasedRolloutFnTrainOutput(
                    samples=cast(list[list[Sample]], data),
                    metrics=metrics,
                    lease=_OwnedTrainBatchLease(
                        rollout_id=input.rollout_id,
                        owner=self,
                        terminal_receipts=terminal_receipts,
                    ),
                )
            return RolloutFnTrainOutput(samples=data, metrics=metrics)
        except BaseException as error:
            if terminal_receipts:
                try:
                    self._rollback_owned_terminals(terminal_receipts)
                except BaseException as settlement_error:
                    self._pending_terminal_rollbacks.extend((receipt, True) for receipt in terminal_receipts)
                    raise error from settlement_error
            raise

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])

    def _handle_unused_buffer_source(self, source: DataBufferSource) -> None:
        if isinstance(source, ReservationTerminalReceipt):
            self._rollback_owned_terminal(source, completed_slot_held=True)
            return
        self._handle_unused(source)

    def _discard_buffer_source(self, source: DataBufferSource) -> None:
        if isinstance(source, ReservationTerminalReceipt):
            self._discarded_terminal_receipts.append(source)
            self._discarded_terminal_available.set()

    def _flush_discarded_terminals(self, rollout_id: int) -> None:
        ownership = self._require_ownership()
        while self._discarded_terminal_receipts:
            receipt = self._discarded_terminal_receipts[0]
            ownership.commit_batch([receipt], rollout_id=rollout_id)
            self._discarded_terminal_receipts.popleft()
            self._release_owned_capacity([receipt], completed_slots=1)
        self._discarded_terminal_available.clear()

    def _commit_owned_terminals(
        self,
        terminal_receipts: list[ReservationTerminalReceipt],
        *,
        rollout_id: int,
    ) -> None:
        self._require_ownership().commit_batch(terminal_receipts, rollout_id=rollout_id)
        self._release_owned_capacity(terminal_receipts, completed_slots=len(terminal_receipts))

    def _rollback_owned_terminal(
        self,
        terminal_receipt: ReservationTerminalReceipt,
        *,
        completed_slot_held: bool,
    ) -> None:
        self._require_ownership().rollback_batch([terminal_receipt])
        self._release_owned_capacity([terminal_receipt], completed_slots=int(completed_slot_held))

    def _rollback_owned_terminals(self, terminal_receipts: list[ReservationTerminalReceipt]) -> None:
        self._require_ownership().rollback_batch(terminal_receipts)
        self._release_owned_capacity(terminal_receipts, completed_slots=len(terminal_receipts))

    def _release_owned_capacity(
        self,
        terminal_receipts: list[ReservationTerminalReceipt],
        *,
        completed_slots: int,
    ) -> None:
        retained_slots = self._require_retained_slots()
        owned_completed_slots = self._completed_slots
        if owned_completed_slots is None:
            raise RuntimeError("Fully async completed capacity is not initialized.")
        for _ in terminal_receipts:
            retained_slots.release()
        for _ in range(completed_slots):
            owned_completed_slots.put_nowait(object())
        if completed_slots:
            self._completed_slot_available.set()
        self._owned_capacity_released.set()

    def _require_ownership(self) -> ReservationOwnership:
        if self._ownership is None:
            raise RuntimeError("Fully async reservation ownership is not initialized.")
        return self._ownership

    def _require_retained_slots(self) -> asyncio.BoundedSemaphore:
        if self._retained_slots is None:
            raise RuntimeError("Fully async retained capacity is not initialized.")
        return self._retained_slots
