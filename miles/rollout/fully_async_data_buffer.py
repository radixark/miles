"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics); ``DefaultDataBuffer``
is the built-in implementation, replaceable via ``--custom-async-data-buffer-path``.
Every group-level decision lives here — what to keep, what to hand to
``--async-unused-samples-handler`` — so a custom buffer owns all of it. Only
``--rollout-sample-filter-path`` stays outside: it runs on the assembled batch.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter, iter_samples
from miles.rollout.filter_hub.common_filters import (
    GroupWeightVersionStats,
    apply_aborted_filter,
    apply_missing_reward_filter,
    group_staleness,
    group_weight_version_stats,
)
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
    completed_at: float | None = None  # monotonic time when generation and reward finished


@dataclass(frozen=True)
class _BufferedEntry:
    input: DataBufferInput
    enqueued_at: float


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

        self._buffer: list[_BufferedEntry] = []
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
        self._metric_selected_newest_lag: list[int] = []
        self._metric_selected_version_span: list[int] = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0

        now = time.monotonic()
        self._metric_window_started_at = now
        self._last_state_change_at = now
        self._metric_queue_size_time = 0.0
        self._metric_queue_empty_time = 0.0
        self._metric_queue_full_time = 0.0
        self._metric_producer_blocked_time = 0.0
        self._metric_consumer_wait_time = 0.0
        self._metric_queue_high_watermark = 0
        self._metric_producer_block_events = 0
        self._metric_consumer_wait_events = 0
        self._metric_selected_queue_residence: list[float] = []
        self._metric_selected_ready_age: list[float] = []
        self._metric_stale_ready_age: list[float] = []
        self._metric_selected_group_tokens: list[int] = []
        self._metric_filtered_group_tokens: list[int] = []
        self._metric_selected_groups = 0

        self._producer_waiters = 0
        self._consumer_waiters = 0
        self._pending_puts = 0
        self._total_received_groups = 0
        self._total_prebuffer_filtered_groups = 0
        self._total_buffered_groups = 0
        self._total_popped_groups = 0
        self._total_selected_groups = 0
        self._total_stale_filtered_groups = 0

    async def put(self, input: DataBufferInput) -> None:
        self._total_received_groups += 1
        if not self._preput_filter(input):
            self._total_prebuffer_filtered_groups += 1
            return

        self._pending_puts += 1
        try:
            async with self._cond:
                if len(self._buffer) >= self._capacity:
                    self._record_state(time.monotonic())
                    self._producer_waiters += 1
                    self._metric_producer_block_events += 1
                    try:
                        while len(self._buffer) >= self._capacity:
                            await self._cond.wait()
                    finally:
                        self._record_state(time.monotonic())
                        self._producer_waiters -= 1

                now = time.monotonic()
                self._record_state(now)
                self._buffer.append(_BufferedEntry(input=input, enqueued_at=now))
                self._metric_queue_high_watermark = max(self._metric_queue_high_watermark, len(self._buffer))
                self._total_buffered_groups += 1
                self._cond.notify_all()
        finally:
            self._pending_puts -= 1

    def _preput_filter(self, input: DataBufferInput) -> bool:
        output = apply_aborted_filter(self._args, input.group)
        if not output.keep:
            self._metric_aborted_groups += 1
            self._record_filtered_group(input.group)
            self._unused_handler_fn(input.prompt_group)
            return False

        output = apply_missing_reward_filter(self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            self._record_filtered_group(input.group)
            return False

        output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            self._record_filtered_group(input.group)
            return False
        return True

    async def get(self, current_version: int | None = None, **_) -> DataBufferInput:
        if current_version is not None:
            self._current_version = current_version
        async with self._cond:
            while True:
                while not self._buffer:
                    self._record_state(time.monotonic())
                    self._consumer_waiters += 1
                    self._metric_consumer_wait_events += 1
                    try:
                        await self._cond.wait()
                    finally:
                        self._record_state(time.monotonic())
                        self._consumer_waiters -= 1
                now = time.monotonic()
                self._record_state(now)
                buffered = self._buffer.pop(0)
                entry = buffered.input
                self._total_popped_groups += 1
                self._cond.notify_all()  # wake producers blocked on a full buffer

                version_stats = group_weight_version_stats(entry.group)
                staleness = version_stats.oldest_lag(current_version)
                if staleness is not None:
                    if self._args.max_weight_staleness is not None and staleness > self._args.max_weight_staleness:
                        logger.info(f"Filtered stale group ({staleness=} > max={self._args.max_weight_staleness})")
                        self._metric_stale_groups += 1
                        self._total_stale_filtered_groups += 1
                        self._record_ready_age(self._metric_stale_ready_age, entry, now)
                        self._record_filtered_group(entry.group)
                        self._unused_handler_fn(entry.prompt_group)
                        continue
                    self._metric_consumed_staleness.append(staleness)
                self._record_selected_version_stats(version_stats, current_version)
                self._metric_selected_groups += 1
                self._total_selected_groups += 1
                self._metric_selected_queue_residence.append(now - buffered.enqueued_at)
                self._record_ready_age(self._metric_selected_ready_age, entry, now)
                self._metric_selected_group_tokens.append(self._group_response_tokens(entry.group))
                return entry

    def _record_state(self, now: float) -> None:
        elapsed = now - self._last_state_change_at
        queue_size = len(self._buffer)
        self._metric_queue_size_time += elapsed * queue_size
        self._metric_queue_empty_time += elapsed * int(queue_size == 0)
        self._metric_queue_full_time += elapsed * int(queue_size >= self._capacity)
        self._metric_producer_blocked_time += elapsed * int(self._producer_waiters > 0)
        self._metric_consumer_wait_time += elapsed * int(self._consumer_waiters > 0)
        self._last_state_change_at = now

    @staticmethod
    def _group_response_tokens(group: Group) -> int:
        return sum(sample.response_length for sample in iter_samples(group))

    def _record_filtered_group(self, group: Group) -> None:
        self._metric_filtered_group_tokens.append(self._group_response_tokens(group))

    @staticmethod
    def _record_ready_age(target: list[float], entry: DataBufferInput, now: float) -> None:
        if entry.completed_at is not None:
            target.append(now - entry.completed_at)

    def _record_selected_version_stats(
        self,
        stats: GroupWeightVersionStats,
        current_version: int | None,
    ) -> None:
        self._metric_selected_samples += stats.sample_count
        self._metric_selected_versioned_samples += stats.versioned_sample_count

        if stats.oldest_version is not None and stats.newest_version is not None:
            self._metric_selected_version_span.append(stats.newest_version - stats.oldest_version)

        newest_lag = stats.newest_lag(current_version)
        if newest_lag is not None:
            self._metric_selected_newest_lag.append(newest_lag)

        token_weighted_lag = stats.token_weighted_lag(current_version)
        if token_weighted_lag is not None:
            self._metric_selected_token_lag_sum += token_weighted_lag * stats.versioned_token_count
            self._metric_selected_versioned_tokens += stats.versioned_token_count

    def get_metrics(self) -> dict[str, float]:
        prefix = "rollout/fully_async/"
        now = time.monotonic()
        self._record_state(now)
        window_seconds = now - self._metric_window_started_at
        metrics = {
            f"{prefix}queue_size": len(self._buffer),
            f"{prefix}queue_capacity": self._capacity,
            f"{prefix}queue_high_watermark": self._metric_queue_high_watermark,
            f"{prefix}producer_block_events": self._metric_producer_block_events,
            f"{prefix}consumer_wait_events": self._metric_consumer_wait_events,
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            f"{prefix}groups_received_total": self._total_received_groups,
            f"{prefix}groups_prebuffer_filtered_total": self._total_prebuffer_filtered_groups,
            f"{prefix}groups_buffered_total": self._total_buffered_groups,
            f"{prefix}groups_popped_total": self._total_popped_groups,
            f"{prefix}groups_selected_total": self._total_selected_groups,
            f"{prefix}groups_stale_filtered_total": self._total_stale_filtered_groups,
            f"{prefix}pending_puts": self._pending_puts,
            **self._metric_gatherer.collect(),
        }
        if window_seconds > 0:
            metrics[f"{prefix}metrics_window_seconds"] = window_seconds
            metrics[f"{prefix}avg_queue_size"] = self._metric_queue_size_time / window_seconds
            metrics[f"{prefix}queue_occupancy_ratio"] = self._metric_queue_size_time / self._capacity / window_seconds
            metrics[f"{prefix}queue_empty_time_ratio"] = self._metric_queue_empty_time / window_seconds
            metrics[f"{prefix}queue_full_time_ratio"] = self._metric_queue_full_time / window_seconds
            metrics[f"{prefix}producer_blocked_time_ratio"] = self._metric_producer_blocked_time / window_seconds
            metrics[f"{prefix}consumer_wait_time_ratio"] = self._metric_consumer_wait_time / window_seconds
            metrics[f"{prefix}selected_groups_per_second"] = self._metric_selected_groups / window_seconds
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        if newest_lag := self._metric_selected_newest_lag:
            metrics[f"{prefix}avg_post_generation_staleness"] = sum(newest_lag) / len(newest_lag)
            metrics[f"{prefix}max_post_generation_staleness"] = max(newest_lag)
        if version_span := self._metric_selected_version_span:
            metrics[f"{prefix}avg_generation_version_span"] = sum(version_span) / len(version_span)
            metrics[f"{prefix}max_generation_version_span"] = max(version_span)
        if self._metric_selected_versioned_tokens:
            metrics[f"{prefix}token_weighted_staleness"] = (
                self._metric_selected_token_lag_sum / self._metric_selected_versioned_tokens
            )
        if self._metric_selected_samples:
            metrics[f"{prefix}weight_version_sample_coverage"] = (
                self._metric_selected_versioned_samples / self._metric_selected_samples
            )
        self._add_mean_max(metrics, f"{prefix}selected_queue_residence_seconds", self._metric_selected_queue_residence)
        self._add_mean_max(metrics, f"{prefix}selected_ready_age_seconds", self._metric_selected_ready_age)
        self._add_mean_max(metrics, f"{prefix}stale_filtered_ready_age_seconds", self._metric_stale_ready_age)
        self._add_mean_max(metrics, f"{prefix}selected_group_response_tokens", self._metric_selected_group_tokens)
        self._add_mean_max(metrics, f"{prefix}filtered_group_response_tokens", self._metric_filtered_group_tokens)
        selected_tokens = sum(self._metric_selected_group_tokens)
        filtered_tokens = sum(self._metric_filtered_group_tokens)
        if decided_tokens := selected_tokens + filtered_tokens:
            metrics[f"{prefix}filtered_response_token_ratio"] = filtered_tokens / decided_tokens
        buffered = [
            s for entry in self._buffer if (s := group_staleness(entry.input.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_selected_newest_lag = []
        self._metric_selected_version_span = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0
        self._metric_window_started_at = now
        self._last_state_change_at = now
        self._metric_queue_size_time = 0.0
        self._metric_queue_empty_time = 0.0
        self._metric_queue_full_time = 0.0
        self._metric_producer_blocked_time = 0.0
        self._metric_consumer_wait_time = 0.0
        self._metric_queue_high_watermark = len(self._buffer)
        self._metric_producer_block_events = 0
        self._metric_consumer_wait_events = 0
        self._metric_selected_queue_residence = []
        self._metric_selected_ready_age = []
        self._metric_stale_ready_age = []
        self._metric_selected_group_tokens = []
        self._metric_filtered_group_tokens = []
        self._metric_selected_groups = 0
        self._metric_aborted_groups = self._metric_stale_groups = 0
        return metrics

    @staticmethod
    def _add_mean_max(metrics: dict[str, float], prefix: str, values: list[int | float]) -> None:
        if values:
            metrics[f"{prefix}/mean"] = sum(values) / len(values)
            metrics[f"{prefix}/max"] = max(values)
