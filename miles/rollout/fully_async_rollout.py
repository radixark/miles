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
    Group,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
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

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or DefaultDataBuffer
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
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                await self._output.put(task.result())

    # -------------------------- consumer --------------------------

    async def _next_group(self, current_version: int | None) -> DataBufferInput:
        queue_get = asyncio.create_task(self._output.get(current_version=current_version))
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
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group(input.weight_version)
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

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics())

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])
