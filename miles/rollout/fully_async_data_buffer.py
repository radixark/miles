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
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


# =================================== shared ===================================


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
# ================================== contract ==================================


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

        ``context`` is the extra information for sample processing at get() time,
        including the ``trainer_model_id`` whose groups are asked for.
        """

    @abstractmethod
    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        """Report the metrics of one policy since its previous call (its window counters reset here)."""


# ============================= one policy buffer ==============================


class DefaultDataBuffer(DataBuffer):
    """FIFO buffer of finished groups, filtering out what training should not see.

    Rejected on put, because the verdict is fixed once the group is generated:

    - aborted groups (the generate function gave up, e.g. an agentic collect timeout)
    - groups ``--dynamic-sampling-filter-path`` does not keep

    Rejected on get, because staleness depends on when the group is consumed:

    - groups beyond ``--max-weight-staleness``

    Dataflow control options:

    (1) capacity: ``--async-data-buffer-capacity-factor`` bounds the buffer at
        floor(factor * rollout_batch_size) groups; when full, put blocks until
        training consumes.
    (2) unused handling: ``--async-unused-samples-handler`` decides what happens
        to aborted and stale groups: drop discards them, retry recycles their
        prompts for regeneration. Dynamic-filter groups are processed per the
        filter's ``keep``.
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
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            self._buffer.append(input)
            self._cond.notify_all()

    async def get(self, current_version: int | None = None, **_) -> DataBufferInput:
        if current_version is not None:
            self._current_version = current_version
        async with self._cond:
            while True:
                while not self._buffer:
                    await self._cond.wait()
                entry = self._buffer.pop(0)
                self._cond.notify_all()  # wake producers blocked on a full buffer

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

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
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
            s for entry in self._buffer if (s := self._staleness(entry.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_aborted_groups = self._metric_stale_groups = 0
        return metrics

    @staticmethod
    def _staleness(group: Group, current_version: int | None) -> int | None:
        oldest = group_oldest_weight_version(group)
        if oldest is None or current_version is None:
            return None
        return current_version - oldest


# ============================ multi policy buffer =============================


class DefaultMultiDataBuffer(DataBuffer):
    """One plain ``DefaultDataBuffer`` per policy model, composed.

    Each policy consumes at its own pace, so each gets its own capacity, staleness accounting and
    metrics, and the single-policy buffer stays untouched.
    """

    def __init__(self, input: DataBufferConstructorInput):
        self._inners: dict[str, DataBuffer] = {
            model_id: DefaultDataBuffer(input) for model_id in resolve_megatron_config(input.args).model_ids
        }

    async def put(self, input: DataBufferInput) -> None:
        # TODO: a full inner blocks the one producer for every policy; give each policy its own dispatcher
        for trainer_model_id, entry in _split_by_trainer_model_id(input).items():
            await self._inner_of(trainer_model_id).put(entry)

    async def get(self, trainer_model_id: str | None = None, **context) -> DataBufferInput:
        return await self._inner_of(trainer_model_id).get(trainer_model_id=trainer_model_id, **context)

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        return self._inner_of(trainer_model_id).get_metrics()

    def _inner_of(self, trainer_model_id: str | None) -> DataBuffer:
        assert trainer_model_id in self._inners, (
            f"trainer_model_id {trainer_model_id!r} trains no policy of this run ({sorted(self._inners)}), so "
            f"its groups would queue up in a buffer nobody drains"
        )
        return self._inners[trainer_model_id]


# TODO: a policy absent from a trajectory shortens its group below n_samples_per_prompt, which the drain refuses
def _split_by_trainer_model_id(input: DataBufferInput) -> dict[str, DataBufferInput]:
    trainer_model_ids = list(dict.fromkeys(sample.trainer_model_id for sample in iter_samples(input.group)))
    assert None not in trainer_model_ids, (
        f"a multi policy run routes every group by the policy it belongs to, so the generate function must stamp "
        f"every sample with its trainer_model_id, but this group carries {trainer_model_ids}"
    )
    return {
        trainer_model_id: DataBufferInput(
            prompt_group=input.prompt_group, group=_filter_group(input.group, trainer_model_id=trainer_model_id)
        )
        for trainer_model_id in trainer_model_ids
    }


def _filter_group(group: Group, *, trainer_model_id: str) -> Group:
    ans: Group = []
    for item in group:
        if isinstance(item, list):
            if kept := [sample for sample in item if sample.trainer_model_id == trainer_model_id]:
                ans.append(kept)
        else:
            if item.trainer_model_id == trainer_model_id:
                ans.append(item)
    return ans
