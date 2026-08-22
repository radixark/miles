"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics / wait_failed / aclose);
``DefaultDataBuffer`` is the built-in implementation, replaceable via
``--custom-async-data-buffer-path``.
Every group-level decision lives here — what to keep, what to hand to
``--async-unused-samples-handler`` — so a custom buffer owns all of it. Only
``--rollout-sample-filter-path`` stays outside: it runs on the assembled batch.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.utils.misc import load_function
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


@dataclass(frozen=True)
# ================================== contract ==================================


class DataBufferConstructorInput:
    args: Namespace
    unused_handler_fn: Callable[[list[Sample]], None]  # --async-unused-samples-handler, applied to unused groups


@dataclass
class DataBufferInput:
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples


def complete_trainer_model_ids(input: DataBufferInput, group_size: int) -> frozenset[str | None]:
    trainer_model_ids = {sample.trainer_model_id for sample in iter_samples(input.group)}
    return frozenset(
        trainer_model_id
        for trainer_model_id in trainer_model_ids
        if len(_filter_group(input.group, trainer_model_id=trainer_model_id)) == group_size
    )


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

    async def wait_failed(self) -> None:
        """Wait until asynchronous buffer work fails, then raise that terminal error."""
        await asyncio.Future()

    async def aclose(self) -> None:
        """Stop background work and wake blocked operations; repeated calls are safe."""
        return None


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
        self._closed_error: RuntimeError | None = None

    async def put(self, input: DataBufferInput) -> None:
        self._raise_if_closed()
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
            self._raise_if_closed()
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
                self._raise_if_closed()
            self._buffer.append(input)
            self._cond.notify_all()

    async def get(self, current_version: int | None = None, **_) -> DataBufferInput:
        if current_version is not None:
            self._current_version = current_version
        async with self._cond:
            while True:
                self._raise_if_closed()
                while not self._buffer:
                    await self._cond.wait()
                    self._raise_if_closed()
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

    async def aclose(self) -> None:
        async with self._cond:
            if self._closed_error is not None:
                return
            self._closed_error = RuntimeError("data buffer is closed")
            self._cond.notify_all()

    def _raise_if_closed(self) -> None:
        if self._closed_error is not None:
            raise self._closed_error

    @staticmethod
    def _staleness(group: Group, current_version: int | None) -> int | None:
        oldest = group_oldest_weight_version(group)
        if oldest is None or current_version is None:
            return None
        return current_version - oldest


# ============================ multi policy buffer =============================


@dataclass(eq=False)
class _DispatchGroup:
    entries: dict[str, DataBufferInput]
    route: frozenset[str]
    remaining: set[str]
    ready: asyncio.Event
    admitted: bool = False


@dataclass
class _OwnedTask:
    cancelled: bool = False


class DefaultMultiDataBuffer(DataBuffer):
    """One plain ``DefaultDataBuffer`` per policy model, composed.

    Each policy consumes at its own pace, so each gets its own capacity, staleness accounting and
    metrics. A fixed dispatcher per policy preserves FIFO order, while an atomic bounded admission
    window lets a complete sibling batch progress before global backpressure stops the producer.
    An asynchronous inner failure is terminal because another sibling may already be admitted.
    """

    def __init__(self, input: DataBufferConstructorInput):
        paths = _parse_data_buffer_paths(input.args.custom_async_data_buffer_path_per_model)
        model_ids = resolve_megatron_config(input.args).model_ids
        assert not (unknown := sorted(set(paths) - set(model_ids))), (
            f"{DATA_BUFFER_PATH_PER_MODEL_FLAG} names {unknown}, which train no policy of this run "
            f"({sorted(model_ids)})"
        )
        self._inners: dict[str, DataBuffer] = {
            model_id: (load_function(paths.get(model_id)) or DefaultDataBuffer)(input) for model_id in model_ids
        }
        self._group_size = input.args.n_samples_per_prompt
        self._dispatch_capacity = input.args.rollout_batch_size
        per_policy_capacity = self._dispatch_capacity * 2 ** (len(model_ids) - 1)
        producer_group_budget = (
            max(1, input.args.async_max_concurrent_samples // self._group_size)
            if input.args.async_max_concurrent_samples is not None
            else self._dispatch_capacity
        )
        self._dispatch_queue_capacity = per_policy_capacity + 2 * producer_group_budget
        self._dispatch_queues = {model_id: deque[_DispatchGroup]() for model_id in model_ids}
        self._ingress_tails: dict[str, asyncio.Future[None] | None] = {model_id: None for model_id in model_ids}
        self._pending_by_route: dict[frozenset[str], list[_DispatchGroup]] = {}
        self._dispatch_tasks: dict[str, asyncio.Task[None]] = {}
        self._failure_tasks: dict[str, asyncio.Task[None]] = {}
        self._condition = asyncio.Condition()
        self._terminal_event = asyncio.Event()
        self._terminal_error: BaseException | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    async def put(self, input: DataBufferInput) -> None:
        assert (
            len(input.group) == self._group_size
        ), f"a generated prompt group must carry {self._group_size} trajectories, got {len(input.group)}"
        entries = _split_by_trainer_model_id(input)
        for trainer_model_id in entries:
            self._inner_of(trainer_model_id)
        complete_entries = {
            trainer_model_id: entry
            for trainer_model_id, entry in entries.items()
            if len(entry.group) == self._group_size
        }
        self._raise_if_terminal()
        if not complete_entries:
            return

        self._ensure_dispatchers()
        route = frozenset(complete_entries)
        dispatch_group = _DispatchGroup(
            entries=complete_entries,
            route=route,
            remaining=set(complete_entries),
            ready=asyncio.Event(),
        )
        registered: bool = False
        tickets_released: bool = False
        loop = asyncio.get_running_loop()
        async with self._condition:
            self._raise_if_terminal()
            previous_tickets = {
                trainer_model_id: self._ingress_tails[trainer_model_id] for trainer_model_id in complete_entries
            }
            ingress_tickets = {trainer_model_id: loop.create_future() for trainer_model_id in complete_entries}
            self._ingress_tails.update(ingress_tickets)
        try:
            await asyncio.gather(
                *(asyncio.shield(ticket) for ticket in previous_tickets.values() if ticket is not None)
            )
            async with self._condition:
                self._raise_if_terminal()
                while any(
                    len(self._dispatch_queues[trainer_model_id]) >= self._dispatch_queue_capacity
                    for trainer_model_id in complete_entries
                ):
                    await self._condition.wait()
                    self._raise_if_terminal()
                for trainer_model_id in complete_entries:
                    self._dispatch_queues[trainer_model_id].append(dispatch_group)
                registered = True
                for ticket in ingress_tickets.values():
                    ticket.set_result(None)
                tickets_released = True
                self._condition.notify_all()

            async with self._condition:
                self._raise_if_terminal()
                while len(self._pending_by_route.get(route, ())) >= self._dispatch_capacity:
                    await self._condition.wait()
                    self._raise_if_terminal()
                self._pending_by_route.setdefault(route, []).append(dispatch_group)
                dispatch_group.admitted = True
        finally:
            if registered:
                async with self._condition:
                    if not dispatch_group.admitted:
                        for trainer_model_id in complete_entries:
                            queue = self._dispatch_queues[trainer_model_id]
                            if dispatch_group in queue:
                                queue.remove(dispatch_group)
                    dispatch_group.ready.set()
                    self._condition.notify_all()
            if not tickets_released:
                for trainer_model_id, ticket in ingress_tickets.items():
                    previous_ticket = previous_tickets[trainer_model_id]
                    if previous_ticket is None or previous_ticket.done():
                        ticket.set_result(None)
                    else:
                        previous_ticket.add_done_callback(lambda _, ticket=ticket: ticket.set_result(None))

    async def get(self, trainer_model_id: str | None = None, **context) -> DataBufferInput:
        inner = self._inner_of(trainer_model_id)
        self._ensure_dispatchers()
        self._raise_if_terminal()
        get_task = asyncio.create_task(inner.get(trainer_model_id=trainer_model_id, **context))
        failure_task = asyncio.create_task(self.wait_failed())
        try:
            await asyncio.wait({get_task, failure_task}, return_when=asyncio.FIRST_COMPLETED)
            self._raise_if_terminal()
            return get_task.result()
        finally:
            for task in (get_task, failure_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(get_task, failure_task, return_exceptions=True)

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        inner_metrics = self._inner_of(trainer_model_id).get_metrics()
        pending = sum(
            trainer_model_id in dispatch_group.remaining
            for dispatch_groups in self._pending_by_route.values()
            for dispatch_group in dispatch_groups
        )
        queue_size = "rollout/fully_async/queue_size"
        inner_metrics[queue_size] = inner_metrics.get(queue_size, 0) + pending
        inner_metrics["rollout/fully_async/dispatch_pending"] = pending
        inner_metrics["rollout/fully_async/dispatch_route_pending"] = sum(
            len(dispatch_groups) for dispatch_groups in self._pending_by_route.values()
        )
        return inner_metrics

    async def wait_failed(self) -> None:
        await self._terminal_event.wait()
        self._raise_if_terminal()
        raise AssertionError("a terminal event must carry an error")

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        terminal_error = self._terminal_error
        self._closing = True
        async with self._condition:
            self._condition.notify_all()

        tasks = [*self._dispatch_tasks.values(), *self._failure_tasks.values()]
        for task in tasks:
            task.cancel()
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        task_errors = [
            result
            for result in task_results
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
        ]
        if terminal_error is None:
            terminal_error = self._terminal_error
        async with self._condition:
            for queue in self._dispatch_queues.values():
                queue.clear()
            self._pending_by_route.clear()
            self._condition.notify_all()

        results = await asyncio.gather(*(inner.aclose() for inner in self._inners.values()), return_exceptions=True)
        errors = [*task_errors, *(result for result in results if isinstance(result, BaseException))]
        if terminal_error is None and errors:
            terminal_error = errors.pop(0)
        self._terminal_error = terminal_error or RuntimeError("data buffer is closed")
        self._terminal_event.set()
        if terminal_error is not None:
            for error in errors:
                logger.error("Additional data buffer close failure", exc_info=error)
            raise terminal_error

    def _ensure_dispatchers(self) -> None:
        self._raise_if_terminal()
        if self._dispatch_tasks:
            return
        self._dispatch_tasks = {
            trainer_model_id: asyncio.create_task(self._dispatch(trainer_model_id))
            for trainer_model_id in self._inners
        }
        self._failure_tasks = {
            trainer_model_id: asyncio.create_task(self._watch_inner(trainer_model_id))
            for trainer_model_id in self._inners
        }

    async def _dispatch(self, trainer_model_id: str) -> None:
        queue = self._dispatch_queues[trainer_model_id]
        inner = self._inners[trainer_model_id]
        while True:
            async with self._condition:
                while not queue:
                    await self._condition.wait()
                dispatch_group = queue.popleft()
                self._condition.notify_all()
            await dispatch_group.ready.wait()
            if not dispatch_group.admitted:
                continue
            ownership = _OwnedTask()
            put_task = asyncio.create_task(
                self._put_inner(
                    inner=inner,
                    input=dispatch_group.entries[trainer_model_id],
                    trainer_model_id=trainer_model_id,
                    ownership=ownership,
                )
            )
            try:
                await asyncio.shield(put_task)
            except asyncio.CancelledError as error:
                if put_task.done():
                    put_task.result()
                if self._closing or self._terminal_error is not None:
                    raise
                await self._fail(self._unexpected_cancellation(trainer_model_id, error))
                return
            except BaseException as error:
                await self._fail(error)
                return
            finally:
                if not put_task.done():
                    ownership.cancelled = True
                    put_task.cancel()
                    [result] = await asyncio.gather(put_task, return_exceptions=True)
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        raise result
                await self._complete(trainer_model_id, dispatch_group)

    async def _watch_inner(self, trainer_model_id: str) -> None:
        ownership = _OwnedTask()
        wait_task = asyncio.create_task(
            self._wait_inner_failure(trainer_model_id=trainer_model_id, ownership=ownership)
        )
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError as error:
            if wait_task.done():
                wait_task.result()
                raise RuntimeError(
                    f"data buffer failure watcher for {trainer_model_id!r} returned normally"
                ) from error
            if self._closing or self._terminal_error is not None:
                raise
            await self._fail(self._unexpected_cancellation(trainer_model_id, error))
        except BaseException as error:
            await self._fail(error)
        else:
            await self._fail(RuntimeError(f"data buffer failure watcher for {trainer_model_id!r} returned normally"))
        finally:
            if not wait_task.done():
                ownership.cancelled = True
                wait_task.cancel()
                [result] = await asyncio.gather(wait_task, return_exceptions=True)
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise result

    async def _put_inner(
        self,
        *,
        inner: DataBuffer,
        input: DataBufferInput,
        trainer_model_id: str,
        ownership: _OwnedTask,
    ) -> None:
        try:
            await inner.put(input)
        except asyncio.CancelledError as error:
            if ownership.cancelled:
                raise
            raise self._unexpected_cancellation(trainer_model_id, error) from error

    async def _wait_inner_failure(self, *, trainer_model_id: str, ownership: _OwnedTask) -> None:
        try:
            await self._inners[trainer_model_id].wait_failed()
        except asyncio.CancelledError as error:
            if ownership.cancelled:
                raise
            raise self._unexpected_cancellation(trainer_model_id, error) from error

    async def _complete(self, trainer_model_id: str, dispatch_group: _DispatchGroup) -> None:
        async with self._condition:
            dispatch_group.remaining.remove(trainer_model_id)
            if not dispatch_group.remaining:
                dispatch_groups = self._pending_by_route[dispatch_group.route]
                dispatch_groups.remove(dispatch_group)
                if not dispatch_groups:
                    del self._pending_by_route[dispatch_group.route]
            self._condition.notify_all()

    async def _fail(self, error: BaseException) -> None:
        if self._terminal_error is not None:
            return
        self._terminal_error = error
        self._terminal_event.set()
        current = asyncio.current_task()
        for task in (*self._dispatch_tasks.values(), *self._failure_tasks.values()):
            if task is not current:
                task.cancel()
        async with self._condition:
            self._condition.notify_all()

    @staticmethod
    def _unexpected_cancellation(trainer_model_id: str, error: asyncio.CancelledError) -> RuntimeError:
        terminal_error = RuntimeError(f"data buffer task for policy {trainer_model_id!r} was cancelled")
        terminal_error.__cause__ = error
        return terminal_error

    def _raise_if_terminal(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._closing:
            raise RuntimeError("data buffer is closed")

    def _inner_of(self, trainer_model_id: str | None) -> DataBuffer:
        assert trainer_model_id in self._inners, (
            f"trainer_model_id {trainer_model_id!r} trains no policy of this run ({sorted(self._inners)}), so "
            f"its groups would queue up in a buffer nobody drains"
        )
        return self._inners[trainer_model_id]


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
    return {
        trainer_model_id: DataBufferInput(
            prompt_group=input.prompt_group, group=_filter_group(input.group, trainer_model_id=trainer_model_id)
        )
        for trainer_model_id in trainer_model_ids
    }


def _filter_group(group: Group, *, trainer_model_id: str | None) -> Group:
    ans: Group = []
    for item in group:
        if isinstance(item, list):
            if kept := [sample for sample in item if sample.trainer_model_id == trainer_model_id]:
                ans.append(kept)
        else:
            if item.trainer_model_id == trainer_model_id:
                ans.append(item)
    return ans
