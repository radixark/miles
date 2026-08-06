"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics); ``DefaultDataBuffer``
is the built-in implementation, replaceable via ``--custom-async-data-buffer-path``.
"""

import asyncio
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from miles.utils.types import Sample

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


@dataclass(frozen=True)
class DataBufferConstructorInput:
    args: Namespace
    recycle_fn: Callable[[list[Sample]], None]  # resets prompts and returns them to the data source


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples
    weight_version: int | None  # engine version when the group finished


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
    """FIFO buffer of finished groups between rollout production and training consumption.

    Dataflow/staleness control options:

    (1) max groups: use ``--async-data-buffer-max-batches`` to set the max size
        of the buffer, in multiples of rollout_batch_size. On overflow the most
        stale groups are evicted; 0 disables eviction and blocks the producer
        when the buffer is full.
    (2) stale handling: ``--async-stale-samples-handler`` decides what happens
        to an evicted group beyond ``--max-weight-staleness``: drop discards
        it, retry recycles its prompts for regeneration. Groups evicted purely
        by capacity are always recycled.
    """

    def __init__(self, input: DataBufferConstructorInput):
        args = input.args
        assert args.async_data_buffer_max_batches >= 0
        assert args.async_stale_samples_handler in ("retry", "drop")
        self._args = args
        self._capacity = (
            args.async_data_buffer_max_batches * args.rollout_batch_size
            if args.async_data_buffer_max_batches
            else 1000  # legacy blocking bound
        )
        self._max_staleness = args.max_weight_staleness
        self._recycle_fn = input.recycle_fn
        # "drop" discards stale groups instead of recycling them
        self._stale_handler_fn = (
            input.recycle_fn if args.async_stale_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._entries: list[DataBufferInput] = []
        self._cond = asyncio.Condition()
        self._latest_weight_version: int | None = None
        self._metric_entered_groups = 0
        self._metric_evicted_stale_groups = 0
        self._metric_evicted_overflow_groups = 0

    async def put(self, input: DataBufferInput) -> None:
        if input.weight_version is not None:
            if self._latest_weight_version is None or input.weight_version > self._latest_weight_version:
                self._latest_weight_version = input.weight_version
        async with self._cond:
            if self._args.async_data_buffer_max_batches > 0:
                self._entries.append(input)
                if len(self._entries) > self._capacity:
                    self._evict_overflow()
            else:
                while len(self._entries) >= self._capacity:
                    await self._cond.wait()
                self._entries.append(input)
            self._metric_entered_groups += 1
            self._cond.notify_all()

    async def get(self) -> DataBufferInput:
        async with self._cond:
            while not self._entries:
                await self._cond.wait()
            entry = self._entries.pop(0)
            self._cond.notify_all()
            return entry

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "queue_size": len(self._entries),
            "evicted_stale_groups": self._metric_evicted_stale_groups,
            "evicted_overflow_groups": self._metric_evicted_overflow_groups,
        }
        if self._metric_entered_groups:
            evicted = self._metric_evicted_stale_groups + self._metric_evicted_overflow_groups
            metrics["evict_rate"] = evicted / self._metric_entered_groups
        if self._latest_weight_version is not None:
            staleness = [
                self._latest_weight_version - oldest
                for entry in self._entries
                if (oldest := group_oldest_weight_version(entry.group)) is not None
            ]
            if staleness:
                metrics["buffer_avg_staleness"] = sum(staleness) / len(staleness)
                metrics["buffer_max_staleness"] = max(staleness)
        self._metric_entered_groups = self._metric_evicted_stale_groups = self._metric_evicted_overflow_groups = 0
        return metrics

    @staticmethod
    def _eviction_key(group: Group) -> tuple[float, float]:
        """Stalest-first sort key: (min, sum) of weight versions; versionless groups rank freshest."""
        versions = [v for s in iter_samples(group) if (v := s.oldest_weight_version) is not None]
        if not versions:
            return (float("inf"), float("inf"))
        return (min(versions), sum(versions))

    def _evict_overflow(self) -> None:
        """Evict stalest-first until nothing is beyond ``max_staleness`` and the buffer fits."""
        while self._entries:
            keys = [self._eviction_key(entry.group) for entry in self._entries]
            index = keys.index(min(keys))
            oldest = group_oldest_weight_version(self._entries[index].group)
            if_exceed_staleness = (
                self._max_staleness is not None
                and self._latest_weight_version is not None
                and oldest is not None
                and self._latest_weight_version - oldest > self._max_staleness
            )
            if not if_exceed_staleness and len(self._entries) <= self._capacity:
                return
            entry = self._entries.pop(index)
            if if_exceed_staleness:
                self._metric_evicted_stale_groups += 1
                self._stale_handler_fn(entry.prompt_group)
            else:
                self._metric_evicted_overflow_groups += 1
                self._recycle_fn(entry.prompt_group)
