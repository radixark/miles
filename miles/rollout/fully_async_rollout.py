"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
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
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples
    weight_version: int | None  # engine version when the group finished


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


class DataBuffer(ABC):
    """Store for finished groups between rollout production and training consumption.

    The producer puts each finished group as it completes; the consumer gets one
    group at a time; get_metrics is collected once per training step. Storage,
    ordering, and dropping are implementation details invisible to callers.
    """

    @abstractmethod
    async def put(self, input: DataBufferInput) -> None:
        """Accept a finished group; may store it, or evict data to make room."""

    @abstractmethod
    async def get(self) -> DataBufferInput:
        """Return one group to train on, waiting until one is available."""

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Report metrics since the previous call (window counters reset on collection)."""


class DefaultDataBuffer(DataBuffer):
    """Finished groups waiting between rollout production and training consumption.

    Supported dataflow/staleness control options:

    (1) max groups: use ``--async-data-buffer-max-batches`` to set the max size
        of the buffer, in multiples of rollout_batch_size. On overflow the most
        stale groups are evicted and their prompts recycled for regeneration;
        0 disables eviction and blocks the producer when the buffer is full.
    (2) order: use ``--async-data-buffer-order`` to set the consumption order,
        fifo (default) or lifo. lifo trains on the freshest group first — pair
        it with (1) and/or ``--max-weight-staleness`` so sunk old groups are
        evicted rather than eventually trained on.
    """

    def __init__(
        self,
        *,
        order: str,
        max_groups: int | None,
        max_staleness: int | None,
        on_evict: Callable[[list[Sample]], None],
    ):
        assert order in ("fifo", "lifo"), f"unknown buffer order: {order}"
        assert max_groups is None or max_groups > 0, f"non-positive buffer capacity: {max_groups}"
        self._order = order
        self._capacity = max_groups if max_groups is not None else OUTPUT_QUEUE_MAX_GROUPS
        self._evict_on_overflow = max_groups is not None
        self._max_staleness = max_staleness
        self._on_evict = on_evict
        self._entries: list[DataBufferInput] = []
        self._cond = asyncio.Condition()
        self._latest_weight_version: int | None = None
        self._entered_groups = 0
        self._evicted_stale_groups = 0
        self._evicted_overflow_groups = 0

    async def put(self, input: DataBufferInput) -> None:
        if input.weight_version is not None:
            self._latest_weight_version = max(self._latest_weight_version or 0, input.weight_version)
        async with self._cond:
            if self._evict_on_overflow:
                self._entries.append(input)
                if len(self._entries) > self._capacity:
                    self._evict_overflow()
            else:
                while len(self._entries) >= self._capacity:
                    await self._cond.wait()
                self._entries.append(input)
            self._entered_groups += 1
            self._cond.notify_all()

    async def get(self) -> DataBufferInput:
        async with self._cond:
            while not self._entries:
                await self._cond.wait()
            entry = self._entries.pop() if self._order == "lifo" else self._entries.pop(0)
            self._cond.notify_all()
            return entry

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "queue_size": len(self._entries),
            "evicted_stale_groups": self._evicted_stale_groups,
            "evicted_overflow_groups": self._evicted_overflow_groups,
        }
        if self._entered_groups:
            evicted = self._evicted_stale_groups + self._evicted_overflow_groups
            metrics["evict_rate"] = evicted / self._entered_groups
        if self._latest_weight_version is not None:
            staleness = [
                self._latest_weight_version - oldest
                for entry in self._entries
                if (oldest := group_oldest_weight_version(entry.group)) is not None
            ]
            if staleness:
                metrics["buffer_avg_staleness"] = sum(staleness) / len(staleness)
                metrics["buffer_max_staleness"] = max(staleness)
        self._entered_groups = self._evicted_stale_groups = self._evicted_overflow_groups = 0
        return metrics

    @staticmethod
    def _eviction_key(group: Group) -> tuple[float, float]:
        """Stalest-first sort key: (min, sum) of weight versions; versionless groups rank freshest."""
        versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
        if not versions:
            return (float("inf"), float("inf"))
        return (min(versions), sum(versions))

    def _evict_overflow(self) -> None:
        """Evict stalest-first until nothing is beyond ``max_staleness`` and the buffer fits."""
        while self._entries:
            keys = [self._eviction_key(entry.group) for entry in self._entries]
            index = keys.index(min(keys))
            # keys[index][0]: stalest group's oldest version, inf if unrecorded
            if_exceed_staleness = (
                self._max_staleness is not None
                and self._latest_weight_version is not None
                and self._latest_weight_version - keys[index][0] > self._max_staleness
            )
            if not if_exceed_staleness and len(self._entries) <= self._capacity:
                return
            if if_exceed_staleness:
                self._evicted_stale_groups += 1
            else:
                self._evicted_overflow_groups += 1
            self._on_evict(self._entries.pop(index).prompt_group)


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router's /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = float("-inf")

    async def get(self, args) -> int | None:
        # Throttles failures too: the drain queries once per group, and an unreachable
        # router would otherwise cost every one of them the full timeout.
        if (time.monotonic() - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            data = await asyncio.wait_for(get(url), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
            self._value = int(data["weight_version"])
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            # Transient router unavailability; the staleness filter is best-effort.
            logger.debug(f"Failed to query engine weight version: {e}")
        finally:
            # Stamped on completion, so a router slower than the TTL still gets throttled.
            self._last_query = time.monotonic()
        return self._value


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            max_batches = self.args.async_data_buffer_max_batches
            self._output = DefaultDataBuffer(
                order=self.args.async_data_buffer_order,
                max_groups=max_batches * self.args.rollout_batch_size if max_batches else None,
                max_staleness=self.args.max_weight_staleness,
                on_evict=self._recycle,
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input.rollout_id)

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

    async def _generate_group(self, prompt_group: list[Sample]) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return prompt_group, result

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                prompt_group, group = task.result()
                version = await self._weight_version.get(self.args)
                await self._output.put(DataBufferInput(prompt_group=prompt_group, group=group, weight_version=version))

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> DataBufferInput:
        queue_get = asyncio.create_task(self._output.get())
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

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        aborted_groups_recycled = 0
        stale_groups_recycled = 0
        staleness_values: list[int] = []
        metric_gatherer = MetricGatherer()
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group()
            group = entry.group
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                self._recycle(entry.prompt_group)
                aborted_groups_recycled += 1
                continue

            oldest = group_oldest_weight_version(group)
            current = await self._weight_version.get(args)
            if oldest is not None and current is not None:
                staleness = current - oldest
                staleness_values.append(staleness)
                if args.max_weight_staleness is not None and staleness > args.max_weight_staleness:
                    self._recycle(entry.prompt_group)
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group (oldest_version={oldest}, current={current}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    continue

            filter_output = call_dynamic_filter(self._dynamic_filter, args, group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            if do_print:
                sample = _first_sample(group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            data.append(group)

        sample = _first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: _first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        metrics = {
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
            **{f"rollout/fully_async/{key}": value for key, value in self._output.get_metrics().items()},
            **metric_gatherer.collect(),
        }
        if staleness_values:
            metrics["rollout/fully_async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
            metrics["rollout/fully_async/max_staleness"] = max(staleness_values)

        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])
