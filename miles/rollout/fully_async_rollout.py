"""Fully asynchronous rollout generation.

A persistent background worker keeps one group-budget wave active and may retain one
additional bounded wave while completed groups are publishing; active and publishing
wrappers together never exceed twice the group budget. Each training step only drains
already-completed groups from the data buffer (see ``fully_async_data_buffer.py``).
Rollout production and training consumption run in parallel, so per-iteration wall
time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.async_utils import run
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._active: set[asyncio.Task] = set()
        self._publishing: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if self._close_task is not None:
            raise RuntimeError("fully-async rollout is closed")
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            default_buffer_cls = (
                DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
            )
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
            self._output = buffer_cls(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
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
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self):
        primary_error: BaseException | None = None
        try:
            while True:
                await self._producer_resumed.wait()
                group_budget = self._max_in_flight_groups()
                wrapper_count = len(self._active) + len(self._publishing)
                while wrapper_count < 2 * group_budget and self._scheduler.has_capacity(
                    pending_groups=len(self._active),
                    group_budget=group_budget,
                    reserved_groups=len(self._publishing),
                ):
                    self._active.add(self._submit_one_group())
                    wrapper_count += 1
                pending = self._active | self._publishing
                if wrapper_count >= 2 * group_budget:
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                else:
                    done, _ = await self._scheduler.wait_for_progress(pending)
                for task in done:
                    if task in self._publishing:
                        self._publishing.remove(task)
                        try:
                            task.result()
                        except asyncio.CancelledError as error:
                            raise RuntimeError("rollout publish task was cancelled") from error
                    else:
                        self._active.remove(task)
                        self._publishing.add(self._submit_publish(task.result()))
        except BaseException as error:
            primary_error = error
            raise
        finally:
            tasks = list(self._active | self._publishing)
            publishing = set(self._publishing)
            cancelled_by_owner: set[asyncio.Task] = set()
            for task in tasks:
                if not task.done() and task.cancel():
                    cancelled_by_owner.add(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._active.clear()
            self._publishing.clear()
            errors: list[BaseException] = []
            for task, result in zip(tasks, results, strict=True):
                if not isinstance(result, BaseException):
                    continue
                if isinstance(result, asyncio.CancelledError):
                    if task in cancelled_by_owner:
                        continue
                    message = (
                        "rollout publish task was cancelled" if task in publishing else "rollout task was cancelled"
                    )
                    cancellation = RuntimeError(message)
                    cancellation.__cause__ = result
                    errors.append(cancellation)
                else:
                    errors.append(result)
            if errors and (primary_error is None or isinstance(primary_error, asyncio.CancelledError)):
                for error in errors[1:]:
                    logger.error("Additional rollout worker cleanup failure", exc_info=error)
                raise errors[0]
            for error in errors:
                logger.error("Additional rollout worker cleanup failure", exc_info=error)

    def _submit_publish(self, input: DataBufferInput) -> asyncio.Task[None]:
        return asyncio.create_task(self._output.put(input))

    # -------------------------- consumer --------------------------

    async def _next_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        queue_get = asyncio.create_task(
            self._output.get(current_version=current_version, trainer_model_id=trainer_model_id)
        )
        buffer_failure = asyncio.create_task(self._output.wait_failed())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, buffer_failure, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if buffer_failure in done:
                    buffer_failure.result()
                    raise RuntimeError("data buffer failure returned without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()
            if not buffer_failure.done():
                buffer_failure.cancel()
            await asyncio.gather(queue_get, buffer_failure, return_exceptions=True)

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group(
                current_version=input.weight_version, trainer_model_id=input.trainer_model_id
            )
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics(input.trainer_model_id))

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        errors: list[BaseException] = []
        if self._worker is not None:
            if not self._worker.done():
                self._worker.cancel()
            [result] = await asyncio.gather(self._worker, return_exceptions=True)
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                errors.append(result)
        if self._output is not None:
            try:
                await self._output.aclose()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                logger.error("Additional fully-async rollout close failure", exc_info=error)
            raise errors[0]

    def dispose(self) -> None:
        run(self.aclose())
