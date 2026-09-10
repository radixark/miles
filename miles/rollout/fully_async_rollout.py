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
from collections import defaultdict, deque
from dataclasses import dataclass


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
    DataBufferState,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    PutOutcome,
    PutOutcomes,
    UnusedReason,
    add_data_buffer_arguments,
    filter_group,
    first_sample,
    iter_samples,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.audit_utils import sample_ownership
from miles.utils.audit_utils.event_logger.models import SampleOwner
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


@dataclass(frozen=True)
class _PendingPrompt:
    samples: list[Sample]
    trainer_model_id: str | None = None


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
        self._retry_buffer: deque[_PendingPrompt] = deque()
        self._in_flight: dict[asyncio.Task, _PendingPrompt] = {}
        self._in_transit: DataBufferState = {}
        self._pending_puts: dict[str | None, DataBufferInput] = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            self._start_worker(weight_version=input.weight_version)
        return await self._drain(input)

    def _start_worker(self, *, weight_version: int | None) -> None:
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
            from_owner = SampleOwner.RETRY_BUFFER
        else:
            [prompt_group] = self.data_source.get_samples(1)
            pending = _PendingPrompt(samples=prompt_group)
            from_owner = SampleOwner.DATA_SOURCE
        self._scheduler.on_submit([pending.samples])
        task = asyncio.create_task(self._generate_group(pending))
        self._in_flight[task] = pending
        sample_ownership.log_owner_transition(
            pending.samples,
            from_owner=from_owner,
            to_owner=SampleOwner.IN_FLIGHT,
            trainer_model_id=pending.trainer_model_id,
        )
        return task

    async def _generate_group(self, pending: _PendingPrompt) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            pending.samples,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        if pending.trainer_model_id is not None:
            result = filter_group(result, trainer_model_id=pending.trainer_model_id)
            assert result, f"Regeneration returned no samples for retry policy {pending.trainer_model_id!r}"
        return DataBufferInput(prompt_group=pending.samples, group=result)

    async def _worker_loop(self) -> None:
        active: set[asyncio.Task] = set()
        await self._flush_pending_puts()
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                entry = task.result()
                self._pending_puts = (
                    self._output.partition(entry)
                    if resolve_megatron_config(self.args).is_multi_policy
                    else {None: entry}
                )
                self._in_flight.pop(task)
                sample_ownership.log_group_routed(entry.prompt_group)
                for model_id, pending in self._pending_puts.items():
                    sample_ownership.log_owner_transition(
                        iter_samples(pending.group),
                        from_owner=SampleOwner.IN_FLIGHT,
                        to_owner=SampleOwner.OUTPUT_BUFFER,
                        trainer_model_id=model_id,
                    )
                await self._flush_pending_puts()

    async def _flush_pending_puts(self) -> None:
        while self._pending_puts:
            model_id = next(iter(self._pending_puts))
            entry = self._pending_puts[model_id]
            await self._output.put(entry)
            assert (
                entry.completed_outcomes is not None
            ), "DataBuffer.put must set input.completed_outcomes when it transfers ownership"
            self._settle_pending_puts()

    def _settle_pending_puts(self) -> None:
        for model_id, entry in list(self._pending_puts.items()):
            if (outcomes := entry.completed_outcomes) is None:
                continue
            del self._pending_puts[model_id]
            self._log_put_outcomes(
                [(entry.group, {model_id if key is None else key: value for key, value in outcomes.items()})]
            )

    def _log_put_outcomes(self, completed: list[tuple[Group, PutOutcomes]]) -> None:
        transitions: dict[tuple[SampleOwner, str | None], list[Sample]] = defaultdict(list)
        for group, outcomes in completed:
            for trainer_model_id, outcome in outcomes.items():
                if outcome is PutOutcome.RECYCLED:
                    continue
                to_owner = SampleOwner.OUTPUT_BUFFER if outcome is PutOutcome.KEPT else SampleOwner.DROPPED
                policy_group = (
                    group if trainer_model_id is None else filter_group(group, trainer_model_id=trainer_model_id)
                )
                transitions[to_owner, trainer_model_id].extend(iter_samples(policy_group))
        for (to_owner, trainer_model_id), samples in transitions.items():
            sample_ownership.log_owner_transition(
                samples,
                from_owner=SampleOwner.IN_FLIGHT,
                to_owner=to_owner,
                trainer_model_id=trainer_model_id,
                reason="dynamic_filter" if to_owner is SampleOwner.DROPPED else None,
            )

    # -------------------------- consumer --------------------------

    async def _take_batch(
        self, *, num_groups: int, current_version: int | None, trainer_model_id: str | None
    ) -> list[DataBufferInput]:
        batch = await self._output.get(
            num_groups=num_groups, current_version=current_version, trainer_model_id=trainer_model_id
        )
        self._in_transit[trainer_model_id] = batch
        return batch

    async def _next_batch(
        self, *, num_groups: int, current_version: int | None, trainer_model_id: str | None
    ) -> list[DataBufferInput]:
        queue_get = asyncio.create_task(
            self._take_batch(num_groups=num_groups, current_version=current_version, trainer_model_id=trainer_model_id)
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

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        entries = await self._next_batch(
            num_groups=args.rollout_batch_size,
            current_version=input.weight_version,
            trainer_model_id=input.trainer_model_id,
        )
        assert len(entries) == args.rollout_batch_size
        for entry in entries:
            assert len(entry.group) == args.n_samples_per_prompt

        data: list[Group] = [entry.group for entry in entries]

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

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        metrics = self._output.get_metrics(input.trainer_model_id)
        self._in_transit.pop(input.trainer_model_id, None)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample], *, reason: UnusedReason, trainer_model_id: str | None) -> None:
        prompt_group = _copy_reset_for_retry(prompt_group)
        self._retry_buffer.append(_PendingPrompt(samples=prompt_group, trainer_model_id=trainer_model_id))
        sample_ownership.log_owner_transition(
            prompt_group,
            from_owner=SampleOwner.IN_FLIGHT if reason is UnusedReason.ABORTED else SampleOwner.OUTPUT_BUFFER,
            to_owner=SampleOwner.RETRY_BUFFER,
            trainer_model_id=trainer_model_id,
            reason=reason.value,
        )

    def _drop_unused(self, prompt_group: list[Sample], *, reason: UnusedReason, trainer_model_id: str | None) -> None:
        sample_ownership.log_owner_transition(
            prompt_group,
            from_owner=SampleOwner.IN_FLIGHT if reason is UnusedReason.ABORTED else SampleOwner.OUTPUT_BUFFER,
            to_owner=SampleOwner.DROPPED,
            trainer_model_id=trainer_model_id,
            reason=reason.value,
        )

    # ------------------------- checkpointing --------------------------

    def describe_holdings(self, trainer_model_id: str | None) -> dict[SampleOwner, list[int]]:
        self._settle_pending_puts()
        return self._holdings(buffered=self._buffered(), trainer_model_id=trainer_model_id)

    def replays_samples(self, trainer_model_id: str | None) -> bool:
        self._ensure_output()
        if isinstance(self._output, DefaultMultiDataBuffer):
            return self._output.replays_samples_of(trainer_model_id)
        return self._output.replays_samples

    def _ensure_output(self) -> None:
        if self._output is not None:
            return
        default_buffer_cls = (
            DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
        )
        buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
        self._output = buffer_cls(DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused))

    def _holdings(self, *, buffered: DataBufferState, trainer_model_id: str | None) -> dict[SampleOwner, list[int]]:
        return {
            SampleOwner.RETRY_BUFFER: [
                s.index
                for pending in self._retry_buffer
                if pending.trainer_model_id in (None, trainer_model_id)
                for s in pending.samples
                if s.index is not None
            ],
            SampleOwner.IN_FLIGHT: [
                s.index
                for pending in self._in_flight.values()
                if pending.trainer_model_id in (None, trainer_model_id)
                for s in pending.samples
                if s.index is not None
            ],
            SampleOwner.OUTPUT_BUFFER: [
                s.index
                for entry in [
                    *buffered.get(trainer_model_id, []),
                    *([pending] if (pending := self._pending_puts.get(trainer_model_id)) is not None else []),
                ]
                for s in iter_samples(entry.group)
                if s.index is not None
            ],
        }

    def _buffered(self) -> DataBufferState:
        return self._merge_output_state(self._output.snapshot() if self._output is not None else {})

    def _merge_output_state(self, buffered: DataBufferState) -> DataBufferState:
        ans: DataBufferState = {key: list(entries) for key, entries in self._in_transit.items()}
        for key, entries in buffered.items():
            ans.setdefault(key, []).extend(entries)
        return ans


def _copy_reset_for_retry(prompt_group: list[Sample]) -> list[Sample]:
    ans = copy.deepcopy(prompt_group)
    _reset_for_retry(ans)
    return ans


def _reset_for_retry(prompt_group: list[Sample]) -> None:
    for sample in prompt_group:
        sample.reset_for_retry()


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
