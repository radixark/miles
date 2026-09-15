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
from dataclasses import dataclass
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
    compute_kv_cache_namespace,
    stamp_kv_cache_namespace,
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
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.function_registry import load_function
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


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
        self._disposed: bool = False
        self._worker_error_cancel_targets: set[asyncio.Task] = set()
        self._eval_prompt_dataset_cache: dict = {}
        self._curr_kv_cache_namespace: str | None = None
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        default_buffer_cls = (
            DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
        )
        buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
        self._output: DataBuffer = buffer_cls(
            DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
        )
        self._retry_buffer: deque[list[Sample]] = deque()
        self._running_tasks: list[_RunningTask] = []

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        self._curr_kv_cache_namespace = compute_kv_cache_namespace(self.args, input)
        if self._worker is None:
            assert input.weight_version is not None, (
                "the orchestration publishes the restored weight version before the producer starts, or every "
                "group a checkpoint restored is filtered out as stale"
            )
            self._worker = asyncio.create_task(self._worker_loop())
            self._worker.add_done_callback(self._on_worker_error)
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        if (worker := self._worker) is None or self._disposed:
            return
        self._disposed = True
        await _end_worker(worker)
        self._log_held_groups_as_dropped()

    def _log_held_groups_as_dropped(self) -> None:
        held = [
            *(x.prompt_group for x in self._running_tasks),
            *self._retry_buffer,
            *self._output.held_prompt_groups(),
        ]
        SampleOwnershipRecorder.log_dropped_samples(
            args=self.args,
            samples=SampleOwnershipRecorder.flatten_samples(held),
            reason="shutdown_in_flight",
        )

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(
                input.generate_state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(
                self.state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
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
        samples = [self._retry_buffer.popleft()] if self._retry_buffer else self.data_source.get_samples(1)
        stamp_kv_cache_namespace(samples, namespace=self._curr_kv_cache_namespace)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        task = asyncio.create_task(self._generate_group(prompt_group))
        self._running_tasks.append(_RunningTask(prompt_group=prompt_group, task=task))
        return task

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
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                await self._output.put(task.result())
                self._running_tasks = [x for x in self._running_tasks if x.task is not task]

    # -------------------------- consumer --------------------------

    def _on_worker_error(self, worker: asyncio.Task) -> None:
        # The worker loop never returns normally, so any completion fails the waiting step.
        logger.error("fully-async rollout worker ended", exc_info=None if worker.cancelled() else worker.exception())
        for target in list(self._worker_error_cancel_targets):
            target.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        assert not self._worker.done(), "the fully-async rollout worker has ended"
        self._worker_error_cancel_targets.add(asyncio.current_task())
        entries = await self._output.get(
            current_version=input.weight_version,
            num_groups=args.rollout_batch_size,
            trainer_model_id=input.trainer_model_id,
        )
        self._worker_error_cancel_targets.discard(asyncio.current_task())

        data: list[Group] = []
        do_print = True

        for entry in entries:
            assert len(entry.group) == args.n_samples_per_prompt

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

        before_filter = SampleOwnershipRecorder.flatten_samples(data)
        if self._sample_filter is not None:
            self._sample_filter(args, data)
        SampleOwnershipRecorder.log_dropped_groups(
            args=args, before=before_filter, after=data, reason="rollout_sample_filter"
        )

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics(input.trainer_model_id))

    def _recycle(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self._retry_buffer.append(prompt_group)

    def _drop_unused(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        SampleOwnershipRecorder.log_dropped_samples(args=self.args, samples=prompt_group, reason=reason.value)

    def save(self, directory: Path) -> None:
        state = _FullyAsyncRolloutState(
            retry_buffer=list(self._retry_buffer),
            running=[_copy_and_reset_for_retry(x.prompt_group) for x in self._running_tasks],
            output=self._output.state_dict(),
        )
        save_simple_checkpoint(directory=directory, data=state)

    def load(self, directory: Path) -> None:
        assert self._worker is None, "restore the fully async rollout state before producer startup"
        state: _FullyAsyncRolloutState = load_simple_checkpoint(directory=directory)
        assert not self._retry_buffer, "restore the fully async rollout state before any group was recycled"
        self._retry_buffer = deque([*state.retry_buffer, *state.running])
        self._output.load_state_dict(state.output)


@dataclass(frozen=True)
class _RunningTask:
    prompt_group: list[Sample]
    task: asyncio.Task


@dataclass(frozen=True)
class _FullyAsyncRolloutState:
    retry_buffer: list[list[Sample]]
    running: list[list[Sample]]
    output: Any


def _copy_and_reset_for_retry(prompt_group: list[Sample]) -> list[Sample]:
    copied = copy.deepcopy(prompt_group)
    for sample in copied:
        sample.reset_for_retry()
    return copied


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
