from __future__ import annotations

import asyncio
from collections.abc import Iterator
from copy import deepcopy
from typing import TypeVar

from miles.rollout.data_source import SourceReservation
from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionOutcome,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncExecutor,
    FullyAsyncRetryReason,
    FullyAsyncTerminalPendingError,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_train import request_abort
from miles.utils.types import Sample

_T = TypeVar("_T")


class _InferenceCancellationCoordinator:
    def __init__(self, state: GenerateState) -> None:
        self._state = state
        self._timeout = state.args.rollout_health_check_timeout
        self._task: asyncio.Task[None] | None = None
        self._executions: set[_InferenceFullyAsyncExecution] = set()

    def register(self, execution: _InferenceFullyAsyncExecution) -> None:
        self._executions.add(execution)
        if self._task is not None:
            execution._observe_cancellation(self._task)

    def unregister(self, execution: _InferenceFullyAsyncExecution) -> None:
        self._executions.discard(execution)

    def request(self) -> asyncio.Task[None]:
        task = self._task
        retry_failed_request = task is not None and task.done() and (task.cancelled() or task.exception() is not None)
        if task is None or retry_failed_request:
            self._state.aborted = True
            task = asyncio.create_task(
                asyncio.wait_for(
                    request_abort(self._state.args),
                    timeout=self._timeout,
                )
            )
            self._task = task
        for execution in tuple(self._executions):
            execution._observe_cancellation(task)
        return task

    def allow_retry_after_terminal_pending(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None

    async def close(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)


class _InferenceFullyAsyncExecution(FullyAsyncExecution):
    def __init__(
        self,
        *,
        cancellation: _InferenceCancellationCoordinator,
        cancellation_timeout: float,
        task: asyncio.Task[list[Sample | list[Sample]]],
        executor_receipt: ReservationExecutorReceipt,
    ) -> None:
        self._cancellation = cancellation
        self._cancellation_timeout = cancellation_timeout
        self._task = task
        self._executor_receipt = executor_receipt
        self._cancellation_task: asyncio.Task[None] | None = None
        self._cancellation_started = asyncio.Event()
        self._cancellation_requested = False
        self._terminal = False
        self._cancellation.register(self)
        self._task.add_done_callback(lambda _task: self._cancellation.unregister(self))

    def request_cancellation(self) -> None:
        if self._terminal or self._task.done():
            return
        self._cancellation.request()

    def _observe_cancellation(self, cancellation_task: asyncio.Task[None]) -> None:
        if self._terminal or self._task.done():
            return
        self._cancellation_requested = True
        self._cancellation_task = cancellation_task
        self._cancellation_started.set()

    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        cancellation_started = await _await_task_terminal(
            asyncio.create_task(
                _wait_for_cancellation_or_terminal(
                    self._task,
                    self._cancellation_started,
                )
            )
        )
        cancellation_error: BaseException | None = None
        cancellation_task: asyncio.Task[None] | None = None
        if cancellation_started:
            cancellation_task = self._cancellation_task
            if cancellation_task is None:
                raise RuntimeError("Fully async inference cancellation has no abort task.")
            try:
                await _await_task_terminal(cancellation_task)
            except BaseException as error:
                cancellation_error = error
        try:
            if cancellation_started:
                samples = await _await_task_terminal_with_timeout(
                    self._task,
                    timeout=self._cancellation_timeout,
                )
            else:
                samples = await _await_task_terminal(self._task)
        except TimeoutError as error:
            if cancellation_task is not None:
                self._cancellation.allow_retry_after_terminal_pending(cancellation_task)
            terminal_error = FullyAsyncTerminalPendingError(
                "Fully async inference cancellation did not make the submitted group "
                f"terminal within {self._cancellation_timeout} seconds."
            )
            if cancellation_error is not None:
                raise terminal_error from cancellation_error
            raise terminal_error from error
        except BaseException as error:
            await self._record_terminal()
            return FullyAsyncExecutionFailure(
                executor_receipt=self._executor_receipt,
                error=error,
            )
        await self._record_terminal()
        for position, item in enumerate(samples):
            parent_samples = item if isinstance(item, list) else [item]
            if any(not isinstance(sample, Sample) for sample in parent_samples):
                return FullyAsyncExecutionFailure(
                    executor_receipt=self._executor_receipt,
                    error=ValueError(f"Fully async inference returned non-Sample values at parent slot {position}."),
                )
        supported_statuses = (
            Sample.Status.PENDING,
            Sample.Status.COMPLETED,
            Sample.Status.TRUNCATED,
            Sample.Status.ABORTED,
            Sample.Status.FAILED,
        )
        for sample in _iter_samples(samples):
            if sample.status not in supported_statuses:
                return FullyAsyncExecutionFailure(
                    executor_receipt=self._executor_receipt,
                    error=ValueError(
                        f"Fully async inference returned sample {sample.index} "
                        f"with unsupported status {sample.status!r}."
                    ),
                )
        if self._cancellation_requested:
            return FullyAsyncExecutionRetry(
                executor_receipt=self._executor_receipt,
                reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
            )
        if any(sample.status in (Sample.Status.PENDING, Sample.Status.ABORTED) for sample in _iter_samples(samples)):
            return FullyAsyncExecutionRetry(
                executor_receipt=self._executor_receipt,
                reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
            )
        return FullyAsyncExecutionSuccess(
            executor_receipt=self._executor_receipt,
            samples=samples,
        )

    async def _record_terminal(self) -> None:
        while self._cancellation_requested:
            cancellation_task = self._cancellation_task
            if cancellation_task is None:
                raise RuntimeError("Fully async inference cancellation has no abort task.")
            try:
                await _await_task_terminal(cancellation_task)
            except BaseException:
                pass
            if self._cancellation_task is not cancellation_task:
                continue
            self._terminal = True
            return
        self._terminal = True


class InferenceFullyAsyncExecutor(FullyAsyncExecutor):
    """Execute receipt-bound inference groups on the caller's event loop."""

    def __init__(self, state: GenerateState) -> None:
        self._state = state
        self._cancellation = _InferenceCancellationCoordinator(state)
        self._tasks: set[asyncio.Task[list[Sample | list[Sample]]]] = set()
        self._closed = False

    def submit(
        self,
        reservation: SourceReservation,
        executor_receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        """Submit a pristine reservation copy for inference.

        Args:
            reservation: Source reservation whose samples remain unchanged.
            executor_receipt: Exact ownership receipt for this execution attempt.

        Returns:
            A terminal-observable execution bound to ``executor_receipt``.

        Raises:
            RuntimeError: The executor is already closed.
            Exception: Submission failed before an inference task was retained.
        """
        if self._closed:
            raise RuntimeError("Fully async inference executor is closed.")
        task = asyncio.create_task(
            _execute_group(
                self._state,
                deepcopy(list(reservation.samples)),
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return _InferenceFullyAsyncExecution(
            cancellation=self._cancellation,
            cancellation_timeout=self._state.args.rollout_health_check_timeout,
            task=task,
            executor_receipt=executor_receipt,
        )

    async def close(self) -> None:
        """Stop accepting groups after submitted inference becomes terminal.

        Returns:
            None after all submitted groups and cancellation requests settle.
        """
        self._closed = True
        tasks = tuple(self._tasks)
        terminal_error: BaseException | None = None
        for task in tasks:
            try:
                await _await_task_terminal(task)
            except BaseException as error:
                if terminal_error is None:
                    terminal_error = error
        try:
            await self._cancellation.close()
        except BaseException as error:
            if terminal_error is None:
                terminal_error = error
        if terminal_error is not None:
            raise terminal_error


async def _execute_group(
    state: GenerateState,
    samples: list[Sample],
) -> list[Sample | list[Sample]]:
    generated_group: list[Sample | list[Sample]] = []
    generated_group.extend(
        await generate_and_rm_group(
            state,
            samples,
            sampling_params=state.sampling_params.copy(),
            evaluation=False,
        )
    )
    return generated_group


def _iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


async def _await_task_terminal_with_timeout(
    task: asyncio.Task[_T],
    *,
    timeout: float,
) -> _T:
    timeout_task = asyncio.create_task(asyncio.wait_for(asyncio.shield(task), timeout=timeout))
    return await _await_task_terminal(timeout_task)


async def _wait_for_cancellation_or_terminal(
    task: asyncio.Task[object],
    cancellation_started: asyncio.Event,
) -> bool:
    cancellation_task = asyncio.create_task(cancellation_started.wait())
    try:
        done, _ = await asyncio.wait(
            (task, cancellation_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        return cancellation_task in done or cancellation_started.is_set()
    finally:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)


async def _await_task_terminal(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
