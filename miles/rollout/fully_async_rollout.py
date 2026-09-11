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
import copy
import logging
from collections import deque
from pathlib import Path
from typing import Any

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
    UnusedReason,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.audit_utils.sample_ownership.flow import log_dropped_groups, log_dropped_samples
from miles.utils.function_registry import load_function
from miles.utils.simple_checkpointer import SimpleCheckpointer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_CHECKPOINTER = SimpleCheckpointer(path_template="rollout/fully_async_state_{rollout_id}.pt")

NO_PROGRESS_WARN_SECS = 30.0


def compute_fully_async_state_path(directory: str | Path, *, rollout_id: int | None) -> Path:
    return _CHECKPOINTER.path(directory, rollout_id=rollout_id)


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
            self._recycle if input.args.async_unused_samples_handler == "retry" else self._drop_unused
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._retry_buffer: deque[list[Sample]] = deque()
        self._in_flight: dict[asyncio.Task, list[Sample]] = {}
        self._pending_outputs: deque[DataBufferInput] = deque()

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            self._start_worker()
        return await self._drain(input)

    def _start_worker(self) -> None:
        self._ensure_output()
        self._worker = asyncio.create_task(self._worker_loop())
        logger.info("Started fully-async rollout worker")

    async def dispose(self) -> None:
        if (worker := self._worker) is None:
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_end_worker(worker), worker.get_loop()))

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
        if self._retry_buffer:
            pending = self._retry_buffer.popleft()
        else:
            [prompt_group] = self.data_source.get_samples(1)
            pending = prompt_group
        self._scheduler.on_submit([pending])
        task = asyncio.create_task(self._generate_group(pending))
        self._in_flight[task] = pending
        return task

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        entry = DataBufferInput(prompt_group=prompt_group, group=result)
        self._in_flight.pop(asyncio.current_task())
        self._pending_outputs.append(entry)
        return entry

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        await self._flush_pending_outputs()
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                task.result()
            await self._flush_pending_outputs()

    async def _flush_pending_outputs(self) -> None:
        while self._pending_outputs:
            entry = self._pending_outputs[0]
            await self._output.put(entry)
            assert entry.completed, "DataBuffer.put must mark the input completed before returning"
            self._pending_outputs.popleft()

    # -------------------------- consumer --------------------------

    async def _next_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        queue_get = asyncio.create_task(
            self._take_group(current_version=current_version, trainer_model_id=trainer_model_id)
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    if self._worker.cancelled():
                        raise RuntimeError("fully-async rollout was disposed while a step waited for groups")
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _take_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        batch = await self._output.get(current_version=current_version, trainer_model_id=trainer_model_id)
        return batch

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        data: list[Group] = []
        while len(data) < args.rollout_batch_size:
            entry = await self._next_group(
                current_version=input.weight_version, trainer_model_id=input.trainer_model_id
            )
            assert len(entry.group) == args.n_samples_per_prompt
            data.append(entry.group)

        sample = first_sample(data[0])
        logger.info(
            "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        before_filter = list(data)
        if self._sample_filter is not None:
            self._sample_filter(args, data)
        log_dropped_groups(
            before_filter,
            data,
            reason="rollout_sample_filter",
            rollout_id=input.rollout_id,
        )

        metrics = self._output.get_metrics(input.trainer_model_id)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample], reason: UnusedReason = UnusedReason.ABORTED) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self._retry_buffer.append(prompt_group)

    def _drop_unused(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        log_dropped_samples(prompt_group, reason=reason.value)

    def save(self, rollout_id: int) -> None:
        _CHECKPOINTER.save(args=self.args, rollout_id=rollout_id, data=self._collect_state())

    def load(self, rollout_id: int | None = None) -> None:
        assert self._worker is None, "restore the fully async rollout state before producer startup"
        if (state := _CHECKPOINTER.load(args=self.args, rollout_id=rollout_id)) is None:
            return
        self._retry_buffer.extend(state["retry_buffer"])
        self._retry_buffer.extend(state["in_flight"])
        self._pending_outputs.extend(state["pending_outputs"])
        self._ensure_output()
        self._output.restore(state["output"])

    def _ensure_output(self) -> None:
        if self._output is not None:
            return
        default_buffer_cls = (
            DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
        )
        buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
        self._output = buffer_cls(DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused))

    def _collect_state(self) -> dict[str, Any]:
        output = {}
        if self._output is not None:
            for key, entries in self._output.snapshot().items():
                output.setdefault(key, []).extend(entries)
        return {
            "retry_buffer": list(self._retry_buffer),
            "in_flight": [_copy_reset_for_retry(pending) for pending in self._in_flight.values()],
            "pending_outputs": [entry for entry in self._pending_outputs if not entry.completed],
            "output": output,
        }


def _copy_reset_for_retry(prompt_group: list[Sample]) -> list[Sample]:
    copied = copy.deepcopy(prompt_group)
    for sample in copied:
        sample.reset_for_retry()
    return copied


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
