"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics / snapshot / restore);
``DefaultDataBuffer`` is the built-in implementation, replaceable via
``--custom-async-data-buffer-path``. Every group-level decision lives here — what to
keep, what to hand to ``--async-unused-samples-handler`` — so a custom buffer owns all
of it. Only ``--rollout-sample-filter-path`` stays outside: it runs on the assembled
batch.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]

DATA_BUFFER_PATH_PER_MODEL_FLAG = "--custom-async-data-buffer-path-per-model"


def add_data_buffer_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        DATA_BUFFER_PATH_PER_MODEL_FLAG,
        type=str,
        nargs="+",
        default=None,
        metavar="MODEL_ID=PATH",
        help=(
            "Per policy form of --custom-async-data-buffer-path, e.g. "
            "--custom-async-data-buffer-path-per-model solver=pkg.SolverBuffer. A run training several "
            "policies composes one buffer per policy (see DefaultMultiDataBuffer); each model id named "
            "here gets that class instead of the built-in one, and every model id left out keeps it. "
            "The model ids are the --megatron-config ones."
        ),
    )


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


# ================================== contract ==================================


class UnusedReason(Enum):
    ABORTED = "aborted"
    STALE = "stale"


class _UnusedHandler(Protocol):
    def __call__(self, prompt_group: list[Sample], *, reason: UnusedReason, trainer_model_id: str | None) -> None: ...


@dataclass(frozen=True)
class DataBufferConstructorInput:
    args: Namespace
    unused_handler_fn: _UnusedHandler  # --async-unused-samples-handler, applied to unused groups


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples
    admission_passed: bool = False
    completed_outcomes: "PutOutcomes | None" = None


class PutOutcome(Enum):
    KEPT = "kept"
    RECYCLED = "recycled"
    DROPPED = "dropped"


DataBufferState = dict[str | None, list[DataBufferInput]]
PutOutcomes = dict[str | None, PutOutcome]


class DataBuffer(ABC):
    """Store for finished groups between rollout production and training consumption.

    The producer puts each finished group as it completes; the consumer gets a whole
    batch at a time; get_metrics is collected once per training step. Storage,
    ordering, and filtering are invisible to callers — an implementation is free
    to reject a group on put, on get, or not at all.
    """

    replays_samples: bool = False

    def partition(self, input: DataBufferInput) -> dict[str | None, DataBufferInput]:
        if all(sample.trainer_model_id is None for sample in iter_samples(input.group)):
            return {None: input}
        return _split_by_trainer_model_id(input)

    @abstractmethod
    async def put(self, input: DataBufferInput) -> PutOutcomes:
        """Accept a finished group; may store it, reject it, or evict to make room."""

    @abstractmethod
    async def get(self, *, num_groups: int, **context: Any) -> list[DataBufferInput]:
        """Return ``num_groups`` groups to train on, waiting until that many are available.

        ``context`` is the extra information for sample processing at get() time,
        including the ``trainer_model_id`` whose groups are asked for.
        """

    @abstractmethod
    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        """Report the metrics of one policy since its previous call (its window counters reset here)."""

    @abstractmethod
    def snapshot(self) -> DataBufferState: ...

    @abstractmethod
    def restore(self, state: DataBufferState) -> None: ...


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
        assert self._capacity >= args.rollout_batch_size, (
            f"--async-data-buffer-capacity-factor {args.async_data_buffer_capacity_factor} bounds the buffer at "
            f"{self._capacity} groups, below the {args.rollout_batch_size} groups a step drains; the producer "
            f"would block on a full buffer while the step waits for a batch that can never be completed"
        )

        self._unused_handler_fn = input.unused_handler_fn
        self._dynamic_filter = load_function(args.dynamic_sampling_filter_path)
        self._cond = asyncio.Condition()
        self._current_version: int | None = None

        self._metric_gatherer = MetricGatherer()
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_consumed_staleness: list[int] = []

    async def put(self, input: DataBufferInput) -> PutOutcomes:
        # filters at receiving sample: abort filter, dynamic filter
        if any(s.status == Sample.Status.ABORTED for s in iter_samples(input.group)):
            self._metric_aborted_groups += 1
            self._unused_handler_fn(input.prompt_group, reason=UnusedReason.ABORTED, trainer_model_id=None)
            input.completed_outcomes = {None: PutOutcome.RECYCLED}
            return input.completed_outcomes
        if not input.admission_passed:
            self._metric_gatherer.on_group_before_dynamic_filter(self._args, input.group)
            filter_output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                self._metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                input.completed_outcomes = {None: PutOutcome.DROPPED}
                return input.completed_outcomes
            input.admission_passed = True

        async with self._cond:
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            self._buffer.append(input)
            input.completed_outcomes = {None: PutOutcome.KEPT}
            self._cond.notify_all()
        return input.completed_outcomes

    async def get(self, *, num_groups: int, current_version: int | None = None, **_: Any) -> list[DataBufferInput]:
        if current_version is not None:
            self._current_version = current_version
        stalenesses: list[int | None] = []
        while True:
            async with self._cond:
                evicted = self._evict_stale(current_version=current_version, stalenesses=stalenesses)
                ready = len(self._buffer) >= num_groups
            for entry in evicted:
                self._unused_handler_fn(entry.prompt_group, reason=UnusedReason.STALE, trainer_model_id=None)
            if ready:
                async with self._cond:
                    ans = self._buffer[:num_groups]
                    del self._buffer[:num_groups]
                    self._cond.notify_all()  # wake producers blocked on a full buffer
                    self._metric_consumed_staleness.extend(s for s in stalenesses[:num_groups] if s is not None)
                    return ans
            async with self._cond:
                await self._cond.wait()

    def snapshot(self) -> DataBufferState:
        return {None: list(self._buffer)}

    def restore(self, state: DataBufferState) -> None:
        assert not self._buffer, f"restore puts back a whole buffer, but this one already holds {len(self._buffer)}"
        self._buffer = [entry for entries in state.values() for entry in entries]

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

    def _evict_stale(self, *, current_version: int | None, stalenesses: list[int | None]) -> list[DataBufferInput]:
        limit = self._args.max_weight_staleness
        scanned = len(stalenesses)
        kept = self._buffer[:scanned]
        evicted: list[DataBufferInput] = []
        for entry in self._buffer[scanned:]:
            staleness = self._staleness(group=entry.group, current_version=current_version)
            if limit is None or staleness is None or staleness <= limit:
                kept.append(entry)
                stalenesses.append(staleness)
                continue
            logger.info(f"Filtered stale group ({staleness=} > max={limit})")
            self._metric_consumed_staleness.append(staleness)
            self._metric_stale_groups += 1
            evicted.append(entry)
        if evicted:
            self._buffer = kept
            self._cond.notify_all()
        return evicted

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
        paths = _parse_data_buffer_paths(input.args.custom_async_data_buffer_path_per_model)
        model_ids = resolve_megatron_config(input.args).model_ids
        assert not (unknown := sorted(set(paths) - set(model_ids))), (
            f"{DATA_BUFFER_PATH_PER_MODEL_FLAG} names {unknown}, which train no policy of this run "
            f"({sorted(model_ids)})"
        )
        self._inners: dict[str, DataBuffer] = {
            model_id: (load_function(paths.get(model_id)) or DefaultDataBuffer)(
                DataBufferConstructorInput(
                    args=input.args,
                    unused_handler_fn=_PolicyUnusedHandler(handler=input.unused_handler_fn, model_id=model_id),
                )
            )
            for model_id in model_ids
        }

    def partition(self, input: DataBufferInput) -> dict[str | None, DataBufferInput]:
        return _split_by_trainer_model_id(input)

    async def put(self, input: DataBufferInput) -> PutOutcomes:
        # TODO: a full inner blocks the one producer for every policy; give each policy its own dispatcher
        outcomes: PutOutcomes = {}
        for trainer_model_id, entry in _split_by_trainer_model_id(input).items():
            [outcome] = (await self._inner_of(trainer_model_id).put(entry)).values()
            outcomes[trainer_model_id] = outcome
        return outcomes

    async def get(
        self, *, num_groups: int, trainer_model_id: str | None = None, **context: Any
    ) -> list[DataBufferInput]:
        return await self._inner_of(trainer_model_id).get(
            num_groups=num_groups, trainer_model_id=trainer_model_id, **context
        )

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        return self._inner_of(trainer_model_id).get_metrics(trainer_model_id=trainer_model_id)

    def replays_samples_of(self, trainer_model_id: str | None) -> bool:
        return self._inner_of(trainer_model_id).replays_samples

    def snapshot(self) -> DataBufferState:
        return {
            model_id: [entry for entries in inner.snapshot().values() for entry in entries]
            for model_id, inner in self._inners.items()
        }

    def restore(self, state: DataBufferState) -> None:
        unknown = sorted(model_id for model_id in state if model_id not in self._inners)
        assert not unknown, (
            f"the checkpoint holds buffered groups of {unknown}, which train no policy of this run "
            f"({sorted(self._inners)}); nothing would ever drain them"
        )
        for model_id, inner in self._inners.items():
            inner.restore({None: state.get(model_id, [])})

    def _inner_of(self, trainer_model_id: str | None) -> DataBuffer:
        assert trainer_model_id in self._inners, (
            f"trainer_model_id {trainer_model_id!r} trains no policy of this run ({sorted(self._inners)}), so "
            f"its groups would queue up in a buffer nobody drains"
        )
        return self._inners[trainer_model_id]


def filter_group(group: Group, *, trainer_model_id: str) -> Group:
    ans: Group = []
    for item in group:
        if isinstance(item, list):
            if kept := [sample for sample in item if sample.trainer_model_id == trainer_model_id]:
                ans.append(kept)
        else:
            if item.trainer_model_id == trainer_model_id:
                ans.append(item)
    return ans


# TODO: a policy absent from a trajectory shortens its group below n_samples_per_prompt, which the drain refuses
def _parse_data_buffer_paths(values: Iterable[str] | None) -> dict[str, str]:
    ans: dict[str, str] = {}
    for value in values or []:
        model_id, separator, path = value.partition("=")
        model_id, path = model_id.strip(), path.strip()
        if not separator or not model_id or not path:
            raise ValueError(f"Invalid {DATA_BUFFER_PATH_PER_MODEL_FLAG} entry {value!r}; expected MODEL_ID=PATH.")
        if model_id in ans:
            raise ValueError(f"Duplicate model id {model_id!r} in {DATA_BUFFER_PATH_PER_MODEL_FLAG}.")
        ans[model_id] = path
    return ans


def _split_by_trainer_model_id(input: DataBufferInput) -> dict[str, DataBufferInput]:
    trainer_model_ids = list(dict.fromkeys(sample.trainer_model_id for sample in iter_samples(input.group)))
    assert None not in trainer_model_ids, (
        f"a multi policy run routes every group by the policy it belongs to, so the generate function must stamp "
        f"every sample with its trainer_model_id, but this group carries {trainer_model_ids}"
    )
    if len(trainer_model_ids) == 1:
        return {trainer_model_ids[0]: input}
    return {
        trainer_model_id: DataBufferInput(
            prompt_group=input.prompt_group, group=filter_group(input.group, trainer_model_id=trainer_model_id)
        )
        for trainer_model_id in trainer_model_ids
    }


@dataclass(frozen=True)
class _PolicyUnusedHandler:
    handler: _UnusedHandler
    model_id: str

    def __call__(self, prompt_group: list[Sample], *, reason: UnusedReason, trainer_model_id: str | None) -> None:
        self.handler(prompt_group, reason=reason, trainer_model_id=self.model_id)
