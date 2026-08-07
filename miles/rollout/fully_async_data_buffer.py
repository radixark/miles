"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics); ``DefaultDataBuffer``
is the built-in implementation, replaceable via ``--custom-async-data-buffer-path``.
Every group-level decision lives here — what to keep, what to evict, what to hand
to ``--async-unused-samples-handler`` — so a custom buffer owns all of it. Only
``--rollout-sample-filter-path`` stays outside: it runs on the assembled batch.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

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
    unused_handler_fn: Callable[[list[Sample]], None]  # --async-unused-samples-handler, applied to unused groups


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples
    weight_version: int | None  # engine version when the group finished


class DataBuffer(ABC):
    """Store for finished groups between rollout production and training consumption.

    The producer puts each finished group as it completes; the consumer gets one
    group at a time, passing the engine version as of now so staleness is judged
    against a live clock; get_metrics is collected once per training step.
    Storage, ordering, and filtering are invisible to callers — an implementation
    is free to reject a group on put, on get, or not at all.
    """

    @abstractmethod
    async def put(self, input: DataBufferInput) -> None:
        """Accept a finished group; may store it, reject it, or evict to make room."""

    @abstractmethod
    async def get(self, current_version: int | None) -> DataBufferInput:
        """Return one group to train on, waiting until one is available."""

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Report fully-qualified metrics since the previous call (window counters reset here)."""


class DefaultDataBuffer(DataBuffer):
    """FIFO buffer of finished groups, filtering out what training should not see.

    Rejected on put, because the verdict is fixed once the group is generated:

    - aborted groups (the generate function gave up, e.g. an agentic collect timeout)
    - groups ``--dynamic-sampling-filter-path`` does not keep

    Rejected on get, because staleness depends on when the group is consumed:

    - groups beyond ``--max-weight-staleness``

    Dataflow control options:

    (1) max groups: use ``--async-data-buffer-capacity-factor`` to set the max
        size of the buffer, floor(factor * rollout_batch_size) groups. On
        overflow the most stale groups are evicted.
    (2) unused handling: ``--async-unused-samples-handler`` decides what happens
        to aborted, stale, and evicted groups: drop discards them, retry
        recycles their prompts for regeneration. Dynamic-filter groups are
        processed per the filter's ``keep``.
    """

    def __init__(self, input: DataBufferConstructorInput):
        args = input.args
        self._args = args

        self._buffer: list[DataBufferInput] = []
        assert args.async_data_buffer_capacity_factor > 0
        self._capacity = int(args.async_data_buffer_capacity_factor * args.rollout_batch_size)
        assert self._capacity >= 1

        self._unused_handler_fn = input.unused_handler_fn
        self._dynamic_filter = load_function(args.dynamic_sampling_filter_path)
        self._cond = asyncio.Condition()
        self._current_version: int | None = None

        self._metric_gatherer = MetricGatherer()
        self._metric_entered_groups = 0
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_evicted_stale_groups = 0
        self._metric_evicted_overflow_groups = 0
        self._metric_consumed_staleness: list[int] = []

    async def put(self, input: DataBufferInput) -> None:
        self._track_version(input.weight_version)

        # filters at receiving sample: abort filter, dynamic filter
        if any(s.status == Sample.Status.ABORTED for s in iter_samples(input.group)):
            self._metric_aborted_groups += 1
            self._unused_handler_fn(input.prompt_group)
            return
        filter_output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not filter_output.keep:
            # Dropped, not recycled: no usable gradient signal.
            self._metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
            return

        async with self._cond:
            self._buffer.append(input)
            if len(self._buffer) > self._capacity:
                self._evict_overflow()
            self._metric_entered_groups += 1
            self._cond.notify_all()

    async def get(self, current_version: int | None) -> DataBufferInput:
        self._track_version(current_version)
        async with self._cond:
            while True:
                while not self._buffer:
                    await self._cond.wait()
                entry = self._buffer.pop(0)

                # filters at retrieving sample: staleness filter
                staleness = self._staleness(entry.group, current_version)
                if staleness is None:
                    return entry
                self._metric_consumed_staleness.append(staleness)
                if self._args.max_weight_staleness is None or staleness <= self._args.max_weight_staleness:
                    return entry
                logger.info(f"Filtered stale group ({staleness=} > max={self._args.max_weight_staleness})")
                self._metric_stale_groups += 1
                self._unused_handler_fn(entry.prompt_group)

    def get_metrics(self) -> dict[str, float]:
        prefix = "rollout/fully_async/"
        metrics = {
            f"{prefix}queue_size": len(self._buffer),
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            f"{prefix}evicted_stale_groups": self._metric_evicted_stale_groups,
            f"{prefix}evicted_overflow_groups": self._metric_evicted_overflow_groups,
            **self._metric_gatherer.collect(),
        }
        if self._metric_entered_groups:
            evicted = self._metric_evicted_stale_groups + self._metric_evicted_overflow_groups
            metrics[f"{prefix}evict_rate"] = evicted / self._metric_entered_groups
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        buffered = [
            s for entry in self._buffer if (s := self._staleness(entry.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_entered_groups = self._metric_aborted_groups = self._metric_stale_groups = 0
        self._metric_evicted_stale_groups = self._metric_evicted_overflow_groups = 0
        return metrics

    def _track_version(self, version: int | None) -> None:
        if version is not None and (self._current_version is None or version > self._current_version):
            self._current_version = version

    @staticmethod
    def _staleness(group: Group, current_version: int | None) -> int | None:
        oldest = group_oldest_weight_version(group)
        if oldest is None or current_version is None:
            return None
        return current_version - oldest

    @staticmethod
    def _eviction_key(group: Group) -> tuple[float, float]:
        """Stalest-first sort key: (min, sum) of weight versions; versionless groups rank freshest."""
        versions = [v for s in iter_samples(group) if (v := s.oldest_weight_version) is not None]
        if not versions:
            return (float("inf"), float("inf"))
        return (min(versions), sum(versions))

    def _evict_overflow(self) -> None:
        """Evict stalest-first until nothing is beyond ``max_staleness`` and the buffer fits."""
        while self._buffer:
            keys = [self._eviction_key(entry.group) for entry in self._buffer]
            index = keys.index(min(keys))
            staleness = self._staleness(self._buffer[index].group, self._current_version)
            if_exceed_staleness = (
                self._args.max_weight_staleness is not None
                and staleness is not None
                and staleness > self._args.max_weight_staleness
            )
            if not if_exceed_staleness and len(self._buffer) <= self._capacity:
                return
            entry = self._buffer.pop(index)
            if if_exceed_staleness:
                self._metric_evicted_stale_groups += 1
            else:
                self._metric_evicted_overflow_groups += 1
            self._unused_handler_fn(entry.prompt_group)
