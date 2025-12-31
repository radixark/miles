import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletedGroup:
    behavior_version: int
    finished_ts: float
    group: list[Sample]


@dataclass(frozen=True)
class StreamingStartParams:
    groups_per_train_step: int
    queue_target: int
    queue_cap: int
    inflight_target: int


def derive_streaming_start_params(args, *, num_engines: int) -> StreamingStartParams:
    groups_per_train_step = args.rollout_batch_size
    queue_target = 2 * groups_per_train_step
    queue_cap = min(4 * groups_per_train_step, num_engines * 16)
    inflight_target = min(3 * groups_per_train_step, num_engines * 8)

    return StreamingStartParams(
        groups_per_train_step=groups_per_train_step,
        queue_target=queue_target,
        queue_cap=queue_cap,
        inflight_target=inflight_target,
    )


class StreamingGroupBuffer:
    def __init__(self, queue_cap: int) -> None:
        self._queue: asyncio.PriorityQueue[tuple[tuple[int, float, int], CompletedGroup]] = asyncio.PriorityQueue(
            maxsize=queue_cap
        )
        self._seq = 0

    def qsize(self) -> int:
        return self._queue.qsize()

    async def put(self, group: CompletedGroup) -> None:
        # Tie-breaker prevents PriorityQueue from comparing CompletedGroup objects when priorities match.
        self._seq += 1
        await self._queue.put(((group.behavior_version, group.finished_ts, self._seq), group))

    async def get(self) -> CompletedGroup:
        _prio, item = await self._queue.get()
        return item


class StreamingRolloutManager:
    def __init__(
        self,
        args,
        data_source,
        *,
        groups_per_train_step: int,
        queue_target: int,
        queue_cap: int,
        inflight_target: int,
        initial_published_version: int = 0,
    ) -> None:
        self.args = args
        self.data_source = data_source

        self.groups_per_train_step = groups_per_train_step
        self.queue_target = queue_target
        self.queue_cap = queue_cap
        self.inflight_target = inflight_target

        self._published_version = initial_published_version
        self.buffer = StreamingGroupBuffer(queue_cap)

        self._stop_event = asyncio.Event()
        self._producer_task: asyncio.Task | None = None
        self._pending: set[asyncio.Task] = set()

        self._produced_groups = 0
        self._consumed_groups = 0
        self._start_ts = time.time()

        self._sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

    def notify_new_version(self, version: int) -> None:
        """Publish the latest trainer weight version for w_first stamping."""
        self._published_version = version

    def start(self) -> None:
        """Start the async producer loop."""
        if self._producer_task is not None:
            return
        self._producer_task = asyncio.create_task(self._producer_loop())

    async def stop(self) -> None:
        """Stop the producer loop and cancel pending generation tasks."""
        self._stop_event.set()

        if self._producer_task is not None:
            self._producer_task.cancel()

        pending = list(self._pending)
        for t in pending:
            t.cancel()

        if self._producer_task is not None:
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass
            self._producer_task = None

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def stats(self) -> dict[str, Any]:
        """Return producer/consumer metrics."""
        elapsed = max(1e-6, time.time() - self._start_ts)
        return {
            "queue_size_groups": self.buffer.qsize(),
            "inflight_groups": len(self._pending),
            "groups_produced_per_s": self._produced_groups / elapsed,
            "groups_consumed_per_s": self._consumed_groups / elapsed,
            "published_version": self._published_version,
        }

    async def get_next_groups(
        self,
        *,
        num_groups: int,
        target_version: int,
    ) -> tuple[list[CompletedGroup], dict[str, Any]]:
        groups: list[CompletedGroup] = []
        staleness_values: list[int] = []
        ages_s: list[float] = []
        empty_wait_s = 0.0

        while len(groups) < num_groups:
            wait_start = time.time()
            item = await self.buffer.get()
            empty_wait_s += time.time() - wait_start

            staleness = target_version - item.behavior_version
            groups.append(item)
            staleness_values.append(staleness)
            ages_s.append(time.time() - item.finished_ts)

        self._consumed_groups += len(groups)
        return groups, {
            "empty_wait_s": empty_wait_s,
            "staleness_values": staleness_values,
            "queue_ages_s": ages_s,
        }

    async def _producer_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.buffer.qsize() >= self.queue_target or len(self._pending) >= self.inflight_target:
                await asyncio.sleep(0.01)
                continue

            groups = self.data_source.get_samples(1)
            if not groups:
                await asyncio.sleep(0.01)
                continue

            group = groups[0]
            behavior_version = self._published_version

            task = asyncio.create_task(
                self._generate_one_group(
                    group=group,
                    behavior_version=behavior_version,
                )
            )
            self._pending.add(task)

            def _done_callback(t: asyncio.Task) -> None:
                self._pending.discard(t)

            task.add_done_callback(_done_callback)

        # Best-effort cleanup
        if self._pending:
            for t in list(self._pending):
                t.cancel()
            await asyncio.gather(*list(self._pending), return_exceptions=True)
            self._pending.clear()

    async def _generate_one_group(
        self,
        *,
        group: list[Sample],
        behavior_version: int,
    ) -> None:
        try:
            from miles.rollout.sglang_rollout import generate_and_rm_group

            group = await generate_and_rm_group(
                self.args,
                group,
                sampling_params=self._sampling_params.copy(),
                evaluation=False,
            )
            for sample in group:
                sample.metadata["weight_version_first"] = behavior_version
            await self.buffer.put(
                CompletedGroup(
                    behavior_version=behavior_version,
                    finished_ts=time.time(),
                    group=group,
                )
            )
            self._produced_groups += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to generate group")
