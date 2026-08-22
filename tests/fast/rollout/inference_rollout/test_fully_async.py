from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from typing import cast

import pytest

import miles.rollout.inference_rollout.fully_async as fully_async_module
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.data_source import SourceReservation, SourceReservationId
from miles.rollout.fully_async.execution import (
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncRetryReason,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.inference_rollout.fully_async import InferenceFullyAsyncExecutor
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.types import Sample


def make_generate_state() -> GenerateState:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        return GenerateFnOutput(samples=sample)

    state = GenerateState.__new__(GenerateState)
    state.args = Namespace(
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        rollout_health_check_timeout=0.1,
        sglang_enable_deterministic_inference=False,
        sglang_router_policy="random",
    )
    state.generate_fn_semaphore = asyncio.Semaphore(1)
    state.sampling_params = {}
    state.generate_function = generate
    state.aborted = False
    return state


async def test_executor_returns_receipt_bound_success_without_mutating_reservation() -> None:
    executor = InferenceFullyAsyncExecutor(make_generate_state())
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-0"),
        samples=(Sample(group_index=0, index=0, prompt="prompt"),),
    )
    executor_receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, executor_receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionSuccess(
        executor_receipt=executor_receipt,
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                reward=1.0,
                status=Sample.Status.COMPLETED,
            )
        ],
    )
    assert reservation == SourceReservation(
        reservation_id=SourceReservationId("source-0"),
        samples=(Sample(group_index=0, index=0, prompt="prompt"),),
    )

    await executor.close()


async def test_executor_retries_terminal_abort_without_mutating_reservation() -> None:
    state = make_generate_state()

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.response = "discarded"
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-1"),
        samples=(Sample(group_index=1, index=10, prompt="prompt"),),
    )
    executor_receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, executor_receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionRetry(
        executor_receipt=executor_receipt,
        reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
    )
    assert reservation == SourceReservation(
        reservation_id=SourceReservationId("source-1"),
        samples=(Sample(group_index=1, index=10, prompt="prompt"),),
    )

    await executor.close()


async def test_executor_returns_receipt_bound_failure_for_unknown_status() -> None:
    state = make_generate_state()

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = cast(Sample.Status, None)
        sample.reward = 0.0
        return GenerateFnOutput(samples=sample)

    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-invalid"),
        samples=(Sample(group_index=2, index=20, prompt="prompt"),),
    )
    executor_receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, executor_receipt)
    outcome = await execution.wait_terminal()

    assert isinstance(outcome, FullyAsyncExecutionFailure)
    assert outcome.executor_receipt is executor_receipt
    assert type(outcome.error) is ValueError
    assert str(outcome.error) == "Fully async inference returned sample 20 with unsupported status None."

    await executor.close()


@pytest.mark.parametrize(
    "generated_samples",
    [
        cast(list[Sample | list[Sample]], [None]),
        cast(list[Sample | list[Sample]], [[None]]),
    ],
    ids=["top-level", "nested"],
)
async def test_executor_returns_receipt_bound_failure_for_non_sample_output(
    monkeypatch: pytest.MonkeyPatch,
    generated_samples: list[Sample | list[Sample]],
) -> None:
    async def generate_and_rm_group(state, samples, sampling_params, evaluation=False):
        return generated_samples

    monkeypatch.setattr(fully_async_module, "generate_and_rm_group", generate_and_rm_group)
    executor = InferenceFullyAsyncExecutor(make_generate_state())
    reservation = SourceReservation(
        reservation_id=SourceReservationId("source-malformed"),
        samples=(Sample(group_index=3, index=30, prompt="prompt"),),
    )
    executor_receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, executor_receipt)
    outcome = await execution.wait_terminal()

    assert isinstance(outcome, FullyAsyncExecutionFailure)
    assert outcome.executor_receipt is executor_receipt
    assert type(outcome.error) is ValueError
    assert str(outcome.error) == "Fully async inference returned non-Sample values at parent slot 0."

    await executor.close()


async def test_repeated_cancellation_waits_for_replacement_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    replacement_abort_started = asyncio.Event()
    release_replacement_abort = asyncio.Event()
    first_abort_observed = asyncio.Event()
    abort_calls = 0

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        generation_started.set()
        await release_generation.wait()
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        return GenerateFnOutput(samples=sample)

    async def request_abort(args: Namespace) -> None:
        nonlocal abort_calls
        abort_calls += 1
        if abort_calls == 1:
            raise RuntimeError("first abort failed")
        replacement_abort_started.set()
        await release_replacement_abort.wait()

    original_await_task_terminal = fully_async_module._await_task_terminal
    first_abort_task: asyncio.Task[None] | None = None

    async def observe_first_abort(task):
        try:
            return await original_await_task_terminal(task)
        finally:
            if task is first_abort_task:
                first_abort_observed.set()

    state = make_generate_state()
    state.generate_function = generate
    monkeypatch.setattr(fully_async_module, "request_abort", request_abort)
    monkeypatch.setattr(fully_async_module, "_await_task_terminal", observe_first_abort)
    executor = InferenceFullyAsyncExecutor(state)
    executor_receipt = cast(ReservationExecutorReceipt, object())
    execution = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-repeated-cancellation"),
            samples=(Sample(group_index=4, index=40, prompt="prompt"),),
        ),
        executor_receipt,
    )

    await generation_started.wait()
    execution.request_cancellation()
    first_abort_task = cast(fully_async_module._InferenceFullyAsyncExecution, execution)._cancellation_task
    assert first_abort_task is not None
    terminal_wait = asyncio.create_task(execution.wait_terminal())
    await first_abort_observed.wait()

    execution.request_cancellation()
    await replacement_abort_started.wait()
    release_generation.set()

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(terminal_wait), timeout=0.01)
    finally:
        release_replacement_abort.set()
        outcome = await terminal_wait
        await executor.close()

    assert outcome == FullyAsyncExecutionRetry(
        executor_receipt=executor_receipt,
        reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
    )
    assert abort_calls == 2


async def test_cancellation_after_generation_finishes_does_not_request_global_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_calls = 0

    async def request_abort(args: Namespace) -> None:
        nonlocal abort_calls
        abort_calls += 1

    state = make_generate_state()
    monkeypatch.setattr(fully_async_module, "request_abort", request_abort)
    executor = InferenceFullyAsyncExecutor(state)
    executor_receipt = cast(ReservationExecutorReceipt, object())
    execution = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-terminal-before-cancellation"),
            samples=(Sample(group_index=5, index=50, prompt="prompt"),),
        ),
        executor_receipt,
    )
    await cast(fully_async_module._InferenceFullyAsyncExecution, execution)._task

    execution.request_cancellation()
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionSuccess(
        executor_receipt=executor_receipt,
        samples=[
            Sample(
                group_index=5,
                index=50,
                prompt="prompt",
                reward=1.0,
                status=Sample.Status.COMPLETED,
            )
        ],
    )
    assert abort_calls == 0
    assert not state.aborted

    await executor.close()


async def test_global_cancellation_retries_every_active_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_count = 0
    all_generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        nonlocal generation_count
        generation_count += 1
        if generation_count == 2:
            all_generation_started.set()
        await release_generation.wait()
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        return GenerateFnOutput(samples=sample)

    async def request_abort(args: Namespace) -> None:
        abort_started.set()
        await release_abort.wait()

    state = make_generate_state()
    state.generate_fn_semaphore = asyncio.Semaphore(2)
    state.generate_function = generate
    monkeypatch.setattr(fully_async_module, "request_abort", request_abort)
    executor = InferenceFullyAsyncExecutor(state)
    first_receipt = cast(ReservationExecutorReceipt, object())
    second_receipt = cast(ReservationExecutorReceipt, object())
    first = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-global-cancellation-first"),
            samples=(Sample(group_index=6, index=60, prompt="first"),),
        ),
        first_receipt,
    )
    second = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-global-cancellation-second"),
            samples=(Sample(group_index=7, index=70, prompt="second"),),
        ),
        second_receipt,
    )
    first_wait = asyncio.create_task(first.wait_terminal())
    second_wait = asyncio.create_task(second.wait_terminal())

    await all_generation_started.wait()
    first.request_cancellation()
    await abort_started.wait()
    release_generation.set()
    await asyncio.gather(
        cast(fully_async_module._InferenceFullyAsyncExecution, first)._task,
        cast(fully_async_module._InferenceFullyAsyncExecution, second)._task,
    )

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(second_wait), timeout=0.01)
    finally:
        release_abort.set()
        first_outcome, second_outcome = await asyncio.gather(first_wait, second_wait)
        await executor.close()

    assert first_outcome == FullyAsyncExecutionRetry(
        executor_receipt=first_receipt,
        reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
    )
    assert second_outcome == FullyAsyncExecutionRetry(
        executor_receipt=second_receipt,
        reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
    )


async def test_executor_close_settles_siblings_before_raising_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions_started = 0
    all_started = asyncio.Event()
    release_failure = asyncio.Event()
    failure_raised = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_finished = asyncio.Event()
    failure = RuntimeError("executor task failed")

    async def generate_and_rm_group(state, samples, sampling_params, evaluation=False):
        nonlocal executions_started
        executions_started += 1
        if executions_started == 2:
            all_started.set()
        if samples[0].index == 40:
            await release_failure.wait()
            failure_raised.set()
            raise failure
        await release_sibling.wait()
        sibling_finished.set()
        return samples

    monkeypatch.setattr(fully_async_module, "generate_and_rm_group", generate_and_rm_group)
    executor = InferenceFullyAsyncExecutor(make_generate_state())
    first_execution = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-4"),
            samples=(Sample(group_index=4, index=40, prompt="prompt"),),
        ),
        cast(ReservationExecutorReceipt, object()),
    )
    second_execution = executor.submit(
        SourceReservation(
            reservation_id=SourceReservationId("source-5"),
            samples=(Sample(group_index=5, index=50, prompt="prompt"),),
        ),
        cast(ReservationExecutorReceipt, object()),
    )
    first_task = cast(fully_async_module._InferenceFullyAsyncExecution, first_execution)._task
    second_task = cast(fully_async_module._InferenceFullyAsyncExecution, second_execution)._task
    monkeypatch.setattr(executor, "_tasks", [first_task, second_task])
    close = asyncio.create_task(executor.close())
    await all_started.wait()

    try:
        release_failure.set()
        await failure_raised.wait()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(close), timeout=0.01)
        assert not sibling_finished.is_set()
    finally:
        release_sibling.set()
        await asyncio.gather(close, return_exceptions=True)

    with pytest.raises(RuntimeError) as error:
        await close

    assert error.value is failure
    assert sibling_finished.is_set()
