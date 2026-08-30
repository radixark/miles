"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics); ``DefaultDataBuffer``
is the built-in implementation, replaceable via ``--custom-async-data-buffer-path``.
Every group-level decision lives here — what to keep, what to hand to
``--async-unused-samples-handler`` — so a custom buffer owns all of it. Only
``--rollout-sample-filter-path`` stays outside: it runs on the assembled batch.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.filter_hub.common_filters import check_no_aborted, check_no_missing_reward, group_staleness
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


@dataclass(frozen=True)
class DataBufferConstructorInput:
    args: Namespace
    unused_handler_fn: Callable[[list[Sample]], None]  # --async-unused-samples-handler, applied to unused groups


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples


class DataBuffer(ABC):
    """Store for finished groups between rollout production and training consumption.

    The producer puts each finished group as it completes; the consumer gets one
    group at a time; get_metrics is collected once per training step. Storage,
    ordering, and filtering are invisible to callers — an implementation is free
    to reject a group on put, on get, or not at all.
    """

    @abstractmethod
    async def put(self, input: DataBufferInput) -> None:
        """Accept a finished group; may store it, reject it, or evict to make room."""

    @abstractmethod
    async def get(self, **context) -> DataBufferInput:
        """Return one group to train on, waiting until one is available.

        ``context`` is the extra information for sample processing at get() time.
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Report fully-qualified metrics since the previous call (window counters reset here)."""


class DefaultDataBuffer(DataBuffer):
    """FIFO buffer of finished groups, filtering out what training should not see.

    Rejected on put, because the verdict is fixed once the group is generated:

    - aborted groups (the generate function gave up, e.g. an agentic collect timeout)
    - groups with a missing reward
    - groups ``--dynamic-sampling-filter-path`` does not keep

    Rejected on get, because staleness depends on when the group is consumed:

    - groups beyond ``--max-weight-staleness``

    Dataflow control options:

    (1) capacity: ``--async-data-buffer-capacity-factor`` bounds the buffer at
        floor(factor * rollout_batch_size) groups; when full, put blocks until
        training consumes.
    (2) unused handling: ``--async-unused-samples-handler`` decides what happens
        to aborted and stale groups: drop discards them, retry recycles their
        prompts for regeneration. Missing-reward and custom-filter rejections
        are discarded directly.
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
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_consumed_staleness: list[int] = []

    async def put(self, input: DataBufferInput) -> None:
        if not self._preput_filter(input):
            return

        async with self._cond:
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            self._buffer.append(input)
            self._cond.notify_all()

    def _preput_filter(self, input: DataBufferInput) -> bool:
        output = check_no_aborted(self._args, input.group)
        if not output.keep:
            self._metric_aborted_groups += 1
            self._unused_handler_fn(input.prompt_group)
            return False

        output = check_no_missing_reward(self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            return False

        output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            return False
        return True

    async def get(self, current_version: int | None = None, **_) -> DataBufferInput:
        if current_version is not None:
            self._current_version = current_version
        async with self._cond:
            while True:
                while not self._buffer:
                    await self._cond.wait()
                entry = self._buffer.pop(0)
                self._cond.notify_all()  # wake producers blocked on a full buffer

                staleness = group_staleness(entry.group, current_version)
                if staleness is not None:
                    self._metric_consumed_staleness.append(staleness)
                    if self._args.max_weight_staleness is not None and staleness > self._args.max_weight_staleness:
                        logger.info(f"Filtered stale group ({staleness=} > max={self._args.max_weight_staleness})")
                        self._metric_stale_groups += 1
                        self._unused_handler_fn(entry.prompt_group)
                        continue
                return entry

    def get_metrics(self) -> dict[str, float]:
        prefix = "rollout/fully_async/"
        metrics = {
            f"{prefix}queue_size": len(self._buffer),
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            **self._metric_gatherer.collect(),
        }
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        buffered = [
            s for entry in self._buffer if (s := group_staleness(entry.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_aborted_groups = self._metric_stale_groups = 0
        return metrics
