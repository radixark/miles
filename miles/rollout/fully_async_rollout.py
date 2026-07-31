"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation is not served by this function; ``--fully-async`` therefore points
``--eval-function-path`` at the standard inference rollout unless it is set
explicitly.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable, Iterator

import httpx

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnInput, RolloutFnOutput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.http_utils import get
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


def _eviction_key(group: Group) -> tuple[float, float]:
    """Sort key ranking groups stalest-first for eviction.

    Ranking by (min weight version, summed weight versions) equals ranking by
    (largest per-sample staleness, largest summed staleness): with equal group
    sizes the current engine version is a constant offset that cancels out.
    Groups with no recorded versions rank freshest — never evicted while a
    versioned group is available.
    """
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    if not versions:
        return (float("inf"), float("inf"))
    return (min(versions), sum(versions))


class GroupBuffer:
    """Finished groups waiting between the producer worker and the training consumer.

    Without ``max_groups`` this is the legacy bounded queue: the producer blocks
    once ``OUTPUT_QUEUE_MAX_GROUPS`` groups wait. With ``max_groups`` set the
    producer never blocks; an overflow evicts to ``on_evict`` (recycling the
    prompts into the data source) — first every group already beyond
    ``max_staleness`` (when configured and the engine version is known), then
    one group at a time by ``_eviction_key``, ties broken randomly.

    ``order`` picks the consumption end: ``"fifo"`` serves the oldest group,
    ``"lifo"`` the freshest. LIFO keeps training near on-policy without
    throttling production, but old groups sink and only leave through eviction
    or a late, stale drain — pair it with ``max_groups``/``max_staleness``.
    """

    def __init__(
        self,
        *,
        order: str,
        max_groups: int | None,
        max_staleness: int | None,
        on_evict: Callable[[Group], None],
    ):
        assert order in ("fifo", "lifo"), f"unknown buffer order: {order}"
        self._order = order
        self._capacity = max_groups if max_groups is not None else OUTPUT_QUEUE_MAX_GROUPS
        self._evict_on_overflow = max_groups is not None
        self._max_staleness = max_staleness
        self._on_evict = on_evict
        self._groups: list[Group] = []
        self._cond = asyncio.Condition()
        self.entered_groups = 0
        self.evicted_stale_groups = 0
        self.evicted_overflow_groups = 0

    @property
    def wants_weight_version(self) -> bool:
        """Whether ``put`` can use ``current_version`` (threshold eviction active)."""
        return self._evict_on_overflow and self._max_staleness is not None

    def qsize(self) -> int:
        return len(self._groups)

    async def put(self, group: Group, *, current_version: int | None = None) -> None:
        async with self._cond:
            if not self._evict_on_overflow:
                while len(self._groups) >= self._capacity:
                    await self._cond.wait()
            self._groups.append(group)
            self.entered_groups += 1
            if self._evict_on_overflow and len(self._groups) > self._capacity:
                self._evict_overflow(current_version)
            self._cond.notify_all()

    async def get(self) -> Group:
        async with self._cond:
            while not self._groups:
                await self._cond.wait()
            group = self._groups.pop() if self._order == "lifo" else self._groups.pop(0)
            self._cond.notify_all()
            return group

    def _evict_overflow(self, current_version: int | None) -> None:
        if self._max_staleness is not None and current_version is not None:
            fresh: list[Group] = []
            stale: list[Group] = []
            for group in self._groups:
                oldest = group_oldest_weight_version(group)
                too_stale = oldest is not None and current_version - oldest > self._max_staleness
                (stale if too_stale else fresh).append(group)
            if stale:
                self._groups = fresh
                self.evicted_stale_groups += len(stale)
                for group in stale:
                    self._on_evict(group)
        while len(self._groups) > self._capacity:
            keys = [_eviction_key(group) for group in self._groups]
            stalest = min(keys)
            index = random.choice([i for i, key in enumerate(keys) if key == stalest])
            self.evicted_overflow_groups += 1
            self._on_evict(self._groups.pop(index))

    def staleness_stats(self, current_version: int | None) -> tuple[float, int] | None:
        """(average, max) staleness across buffered groups, or None when unknown."""
        if current_version is None:
            return None
        values = [
            current_version - oldest
            for group in self._groups
            if (oldest := group_oldest_weight_version(group)) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values), max(values)

    def reset_counters(self) -> None:
        self.entered_groups = 0
        self.evicted_stale_groups = 0
        self.evicted_overflow_groups = 0


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router's /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            data = await asyncio.wait_for(get(url), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            # Transient router unavailability; the staleness filter is best-effort.
            logger.debug(f"Failed to query engine weight version: {e}")
            return self._value
        self._value = int(data["weight_version"])
        self._last_query = now
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
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._output: GroupBuffer | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            raise ValueError(
                "FullyAsyncRolloutFn does not serve eval; set --eval-function-path to "
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            )
        if self._worker is None:
            self._output = GroupBuffer(
                order=self.args.async_buffer_order,
                max_groups=self.args.async_buffer_max_groups,
                max_staleness=self.args.max_weight_staleness,
                on_evict=self._recycle,
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input.rollout_id)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        [group] = self.data_source.get_samples(1)
        return asyncio.create_task(
            generate_and_rm_group(
                self.state,
                group,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
            )
        )

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        while True:
            while len(active) < self._max_in_flight_groups():
                active.add(self._submit_one_group())
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Without a capacity this blocks when the queue is full, pausing
                # submission instead of growing the queue unboundedly; with
                # --async-buffer-max-groups the buffer evicts by staleness instead.
                version = await self._weight_version.get(self.args) if self._output.wants_weight_version else None
                await self._output.put(task.result(), current_version=version)

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> Group:
        queue_get = asyncio.create_task(self._output.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                if queue_get in done:
                    return queue_get.result()
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                logger.warning(
                    f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})"
                )
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
        do_print = True

        while len(data) < target_data_size:
            group = await self._next_group()
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                self._recycle(group)
                aborted_groups_recycled += 1
                continue

            oldest = group_oldest_weight_version(group)
            current = await self._weight_version.get(args)
            if oldest is not None and current is not None:
                staleness = current - oldest
                staleness_values.append(staleness)
                if args.max_weight_staleness is not None and staleness > args.max_weight_staleness:
                    self._recycle(group)
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group (oldest_version={oldest}, current={current}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            data.append(group)

        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

        buffer = self._output
        metrics = {
            "rollout/fully_async/queue_size": buffer.qsize(),
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
            "rollout/fully_async/evicted_stale_groups": buffer.evicted_stale_groups,
            "rollout/fully_async/evicted_overflow_groups": buffer.evicted_overflow_groups,
        }
        if buffer.entered_groups:
            evicted = buffer.evicted_stale_groups + buffer.evicted_overflow_groups
            metrics["rollout/fully_async/evict_rate"] = evicted / buffer.entered_groups
        buffer.reset_counters()
        if staleness_values:
            metrics["rollout/fully_async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
            metrics["rollout/fully_async/max_staleness"] = max(staleness_values)
        if (stats := buffer.staleness_stats(await self._weight_version.get(args))) is not None:
            (
                metrics["rollout/fully_async/buffer_avg_staleness"],
                metrics["rollout/fully_async/buffer_max_staleness"],
            ) = stats

        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, group: Group) -> None:
        for sample in _iter_samples(group):
            sample.reset_for_retry()
        self.data_source.add_samples([group])
