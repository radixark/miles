import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    engine_idx: int
    url: str
    current_version: int
    drain_only: bool = False
    inflight_groups: int = 0
    healthy: bool = True
    consecutive_failures: int = 0


@dataclass
class CompletedGroup:
    behavior_version: int
    finished_ts: float
    engine_idx: int
    group: list[Sample]


@dataclass(frozen=True)
class StreamingStartParams:
    groups_per_train_step: int
    queue_target: int
    queue_cap: int
    inflight_target: int
    min_active_engines: int
    supports_subset_engine_updates: bool


def derive_streaming_start_params(args, *, num_engines: int) -> StreamingStartParams:
    groups_per_train_step = args.rollout_batch_size
    queue_target = 2 * groups_per_train_step
    queue_cap = min(4 * groups_per_train_step, num_engines * 16)
    inflight_target = min(3 * groups_per_train_step, num_engines * 8)

    weight_update_mode = getattr(args, "streaming_async_weight_update_mode", "rolling_drain")

    # rolling_drain keeps most engines active while updating at most one drained engine at a time.
    if weight_update_mode == "rolling_drain":
        min_active_engines = max(num_engines - 1, 1)
    else:
        # Placeholder for future modes; keep everyone routable by default.
        min_active_engines = max(num_engines, 1)

    supports_subset_engine_updates = weight_update_mode == "rolling_drain" and num_engines >= 2

    return StreamingStartParams(
        groups_per_train_step=groups_per_train_step,
        queue_target=queue_target,
        queue_cap=queue_cap,
        inflight_target=inflight_target,
        min_active_engines=min_active_engines,
        supports_subset_engine_updates=supports_subset_engine_updates,
    )


class EnginePool:
    def __init__(
        self,
        engine_urls: list[str],
        *,
        initial_version: int,
        max_consecutive_failures: int = 3,
    ) -> None:
        if any(url is None for url in engine_urls):
            raise ValueError("engine_urls must not contain None")

        self.engines = [
            EngineState(engine_idx=i, url=url, current_version=initial_version)
            for i, url in enumerate(engine_urls)
        ]
        self.max_consecutive_failures = max_consecutive_failures
        self._rr_cursor = 0

    def get_active_engines(self) -> list[EngineState]:
        return [e for e in self.engines if e.healthy and not e.drain_only]

    def get_healthy_engines(self) -> list[EngineState]:
        return [e for e in self.engines if e.healthy]

    def select_engine(self) -> EngineState | None:
        active = self.get_active_engines()
        if not active:
            return None

        # Round-robin over active engines for a stable spread.
        start = self._rr_cursor
        for i in range(len(active)):
            idx = (start + i) % len(active)
            engine = active[idx]
            self._rr_cursor = (idx + 1) % len(active)
            return engine

        return None

    def inc_inflight(self, engine_idx: int) -> None:
        self.engines[engine_idx].inflight_groups += 1

    def dec_inflight(self, engine_idx: int) -> None:
        engine = self.engines[engine_idx]
        engine.inflight_groups = max(0, engine.inflight_groups - 1)

    def mark_failure(self, engine_idx: int) -> None:
        engine = self.engines[engine_idx]
        engine.consecutive_failures += 1
        if engine.consecutive_failures >= self.max_consecutive_failures:
            engine.healthy = False

    def mark_success(self, engine_idx: int) -> None:
        engine = self.engines[engine_idx]
        engine.consecutive_failures = 0
        engine.healthy = True

    def summary(self) -> dict[str, Any]:
        versions: dict[int, int] = {}
        for e in self.engines:
            versions[e.current_version] = versions.get(e.current_version, 0) + 1
        return {
            "num_engines": len(self.engines),
            "num_healthy": sum(1 for e in self.engines if e.healthy),
            "num_active": sum(1 for e in self.engines if e.healthy and not e.drain_only),
            "num_drain_only": sum(1 for e in self.engines if e.healthy and e.drain_only),
            "inflight_groups": sum(e.inflight_groups for e in self.engines),
            "versions": versions,
        }


class StreamingWeightUpdatePolicy(Protocol):
    def on_new_version(self, version: int) -> None: ...

    def get_update_candidates(self) -> list[int]: ...

    def mark_updated(self, engine_indices: list[int], version: int) -> None: ...


class RollingDrainPolicy:
    """Rolling drain-only updates.

    - Multi-engine: keep at least `min_active_engines` routable; mark outdated engines as drain-only
      one-by-one as capacity allows and update at most one idle drained engine at a time.
    - Single-engine: gate admissions by marking the engine drain-only on each new version; the trainer
      performs a global update and then clears drain-only via `mark_updated`.
    """

    def __init__(self, engine_pool: EnginePool, *, min_active_engines: int) -> None:
        self.engine_pool = engine_pool
        self.min_active_engines = min_active_engines
        self._target_version = self._infer_initial_target_version()

    def _infer_initial_target_version(self) -> int:
        if not self.engine_pool.engines:
            return 0
        return max(e.current_version for e in self.engine_pool.engines)

    def on_new_version(self, version: int) -> None:
        self._target_version = version

        # With a single engine, we can't keep capacity while draining. Instead,
        # stop routing new groups until the trainer finishes the global update.
        if len(self.engine_pool.engines) <= 1:
            for e in self.engine_pool.engines:
                e.drain_only = True
            return

        self._apply_drain_policy()

    def get_update_candidates(self) -> list[int]:
        # Single-engine: trainer uses global update (no subset candidates).
        if len(self.engine_pool.engines) <= 1:
            return []

        candidates = [
            e
            for e in self.engine_pool.engines
            if e.healthy and e.drain_only and e.inflight_groups == 0 and e.current_version < self._target_version
        ]
        if not candidates:
            return []

        # Conservative: update at most one engine at a time.
        candidates.sort(key=lambda e: (e.current_version, e.engine_idx))
        return [candidates[0].engine_idx]

    def mark_updated(self, engine_indices: list[int], version: int) -> None:
        for idx in engine_indices:
            engine = self.engine_pool.engines[idx]
            engine.current_version = version
            engine.drain_only = False

        if len(self.engine_pool.engines) <= 1:
            return

        self._apply_drain_policy()

    def _apply_drain_policy(self) -> None:
        # Keep at least `min_active_engines` active. Mark outdated engines as drain-only
        # one-by-one as capacity allows.
        while True:
            active = self.engine_pool.get_active_engines()
            if len(active) <= self.min_active_engines:
                return

            outdated = [e for e in active if e.current_version < self._target_version]
            if not outdated:
                return

            outdated.sort(key=lambda e: (e.inflight_groups, e.current_version, e.engine_idx))
            outdated[0].drain_only = True


def make_streaming_weight_policy(
    weight_update_mode: str,
    *,
    engine_pool: EnginePool,
    min_active_engines: int,
) -> StreamingWeightUpdatePolicy:
    if weight_update_mode == "rolling_drain":
        return RollingDrainPolicy(engine_pool, min_active_engines=min_active_engines)
    raise ValueError(f"Unknown streaming weight update mode: {weight_update_mode}")


class StreamingGroupBuffer:
    def __init__(self, queue_cap: int) -> None:
        self._queue: asyncio.PriorityQueue[tuple[tuple[int, float], CompletedGroup]] = asyncio.PriorityQueue(
            maxsize=queue_cap
        )

    def qsize(self) -> int:
        return self._queue.qsize()

    async def put(self, group: CompletedGroup) -> None:
        await self._queue.put(((group.behavior_version, group.finished_ts), group))

    async def get(self) -> CompletedGroup:
        _prio, item = await self._queue.get()
        return item


class StreamingRolloutManager:
    def __init__(
        self,
        args,
        data_source,
        *,
        engine_urls: list[str],
        groups_per_train_step: int,
        queue_target: int,
        queue_cap: int,
        inflight_target: int,
        min_active_engines: int,
        weight_update_mode: str,
    ) -> None:
        self.args = args
        self.data_source = data_source

        self.groups_per_train_step = groups_per_train_step
        self.queue_target = queue_target
        self.queue_cap = queue_cap
        self.inflight_target = inflight_target

        self.engine_pool = EnginePool(engine_urls, initial_version=0)
        self.weight_policy = make_streaming_weight_policy(
            weight_update_mode,
            engine_pool=self.engine_pool,
            min_active_engines=min_active_engines,
        )
        self.buffer = StreamingGroupBuffer(queue_cap)

        self._stop_event = asyncio.Event()
        self._producer_task: asyncio.Task | None = None
        self._pending: set[asyncio.Task] = set()

        self._produced_groups = 0
        self._consumed_groups = 0
        self._stale_dropped_groups = 0
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
        self.weight_policy.on_new_version(version)

    def get_update_candidates(self) -> list[int]:
        return self.weight_policy.get_update_candidates()

    def mark_engines_updated(self, engine_indices: list[int], version: int) -> None:
        self.weight_policy.mark_updated(engine_indices, version)

    def start(self) -> None:
        if self._producer_task is not None:
            return
        self._producer_task = asyncio.create_task(self._producer_loop())

    async def stop(self) -> None:
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
        elapsed = max(1e-6, time.time() - self._start_ts)
        return {
            "queue_size_groups": self.buffer.qsize(),
            "inflight_groups": self.engine_pool.summary()["inflight_groups"],
            "groups_produced_per_s": self._produced_groups / elapsed,
            "groups_consumed_per_s": self._consumed_groups / elapsed,
            "stale_groups_dropped": self._stale_dropped_groups,
            "engine_pool": self.engine_pool.summary(),
        }

    async def get_next_groups(
        self,
        *,
        num_groups: int,
        target_version: int,
        max_staleness_versions: int,
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
            if staleness > max_staleness_versions:
                self._stale_dropped_groups += 1
                continue

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

            engine = self.engine_pool.select_engine()
            if engine is None:
                await asyncio.sleep(0.05)
                continue

            groups = self.data_source.get_samples(1)
            if not groups:
                await asyncio.sleep(0.01)
                continue

            group = groups[0]
            behavior_version = engine.current_version

            self.engine_pool.inc_inflight(engine.engine_idx)

            task = asyncio.create_task(
                self._generate_one_group(
                    group=group,
                    engine_idx=engine.engine_idx,
                    engine_url=engine.url,
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
        engine_idx: int,
        engine_url: str,
        behavior_version: int,
    ) -> None:
        try:
            from miles.rollout.sglang_rollout import generate_and_rm_group

            group = await generate_and_rm_group(
                self.args,
                group,
                sampling_params=self._sampling_params.copy(),
                evaluation=False,
                base_url=engine_url,
            )
            self.engine_pool.mark_success(engine_idx)
            await self.buffer.put(
                CompletedGroup(
                    behavior_version=behavior_version,
                    finished_ts=time.time(),
                    engine_idx=engine_idx,
                    group=group,
                )
            )
            self._produced_groups += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"Failed to generate group on engine {engine_idx} ({engine_url})")
            self.engine_pool.mark_failure(engine_idx)
        finally:
            self.engine_pool.dec_inflight(engine_idx)
