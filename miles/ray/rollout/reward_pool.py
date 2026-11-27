"""The rollout's handle on one reward pool: batches go round robin to its workers, and a worker found dead is
restarted and sent the same batch again.

Nothing else restarts a reward cell: the worker manager's liveness scan only tears a dead cell down.
"""

import asyncio
import itertools
import logging

from miles.ray.specs.reward import reward_cell_id, reward_pool_id, reward_worker_name
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)

_MAX_RESTARTS_PER_CALL = 3
# A restarted worker loads its scorer before it answers, which for a 7B checkpoint takes minutes on a cold cache.
_READY_TIMEOUT_SECONDS = 1800.0


class RewardPoolClient:
    def __init__(
        self,
        *,
        config: RewardPoolConfig,
        capability: BackendCapability,
        max_restarts_per_call: int = _MAX_RESTARTS_PER_CALL,
        ready_timeout: float = _READY_TIMEOUT_SECONDS,
    ) -> None:
        self._config = config
        self._capability = capability
        self._max_restarts_per_call = max_restarts_per_call
        self._ready_timeout = ready_timeout
        self._handles: dict[int, BaseWorkerHandle] = {}
        self._restart_locks = [asyncio.Lock() for _ in range(config.num_workers)]
        # A call that saw a worker fail restarts it only if no other call restarted it since that call began.
        self._num_restarts = [0] * config.num_workers
        self._next_worker = 0
        # Coalescing: items waiting for a batch, the timer that sends a partial one, and batches in flight.
        self._pending: list[tuple[object, str, asyncio.Future]] = []
        self._flush_timer: asyncio.TimerHandle | None = None
        self._sends: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return self._config.name

    async def score(self, outputs: list, prompts: list[str]) -> list[float | None]:
        """One score per output, in input order; a scorer's None, for an output it could not score, stays None so the
        missing-reward filter drops the sample's group.

        Each worker call holds at most `batch_size` outputs. With `max_wait_ms` > 0, outputs from concurrent calls
        share batches: a batch is sent once full, or `max_wait_ms` after its first output arrived.
        """
        assert len(outputs) == len(prompts), f"{len(outputs)} outputs for {len(prompts)} prompts"
        if self._config.max_wait_ms == 0:
            size = self._config.batch_size
            chunks = [(outputs[i : i + size], prompts[i : i + size]) for i in range(0, len(outputs), size)]
            results = await asyncio.gather(*(self._score_chunk(*chunk) for chunk in chunks))
            return [score for chunk in results for score in chunk]
        loop = asyncio.get_running_loop()
        futures = []
        for output, prompt in zip(outputs, prompts, strict=True):
            futures.append(loop.create_future())
            self._pending.append((output, prompt, futures[-1]))
            if len(self._pending) == 1:
                self._flush_timer = loop.call_later(self._config.max_wait_ms / 1000, self._flush)
            if len(self._pending) == self._config.batch_size:
                self._flush()
        return list(await asyncio.gather(*futures))

    def _flush(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        batch, self._pending = self._pending, []
        task = asyncio.get_running_loop().create_task(self._send(batch))
        self._sends.add(task)
        task.add_done_callback(self._sends.discard)

    async def _send(self, batch: list[tuple[object, str, asyncio.Future]]) -> None:
        outputs, prompts, futures = zip(*batch, strict=True)
        try:
            scores = await self._score_chunk(list(outputs), list(prompts))
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            return
        for future, score in zip(futures, scores, strict=True):
            if not future.done():
                future.set_result(score)

    async def _score_chunk(self, outputs: list, prompts: list[str]) -> list[float | None]:
        scores = await self._call(self._take_next_worker(), "score_batch", outputs, prompts)
        if len(scores) != len(outputs):
            raise ValueError(f"reward pool {self.name} returned {len(scores)} scores for {len(outputs)} outputs")
        return [None if score is None else float(score) for score in scores]

    def _take_next_worker(self) -> int:
        index = self._next_worker % self._config.num_workers
        self._next_worker += 1
        return index

    async def _call(self, index: int, method: str, *args):
        for attempt in itertools.count():
            seen_restarts = self._num_restarts[index]
            try:
                handle = await self._handle(index)
                return await getattr(handle, method)(*args)  # config-access-exempt: worker method chosen by caller
            except WorkerUnreachableError:
                if attempt >= self._max_restarts_per_call:
                    raise
                logger.warning(
                    f"Reward worker {reward_worker_name(self.name, cell_index=index)} is unreachable in {method}; "
                    f"restarting it and calling it again (restart {attempt + 1}/{self._max_restarts_per_call})",
                    exc_info=True,
                )
                await self._restart(index, seen_restarts=seen_restarts)
        raise AssertionError("unreachable")

    async def _handle(self, index: int) -> BaseWorkerHandle:
        if (handle := self._handles.get(index)) is None:
            handle = await self._lookup_handle(index)
            await handle.wait_ready(timeout=self._ready_timeout)
            self._handles[index] = handle
        return handle

    async def _lookup_handle(self, index: int) -> BaseWorkerHandle:
        provider = self._capability.static_worker_provider(pool_id=reward_pool_id(self.name))
        name = reward_worker_name(self.name, cell_index=index)
        # The Ray provider's lookup blocks on ray.get.
        (infos,) = await asyncio.to_thread(
            provider.get_worker_infos, cell_ids=[reward_cell_id(self.name, cell_index=index)]
        )
        try:
            handles = await asyncio.to_thread(provider.get_handles_of_worker_infos, infos)
        except AssertionError as e:
            # RayWorkerManager.get_actor_handle asserts when the cell went down or restarted after the infos were read.
            raise WorkerUnreachableError(f"Reward worker {name} changed while it was looked up: {e}") from e
        if (handle := handles.get(name)) is None:
            # The liveness scan tore the cell down, so the manager lists no worker for it.
            raise WorkerUnreachableError(f"Reward worker {name} has no live actor")
        return handle

    async def _restart(self, index: int, *, seen_restarts: int) -> None:
        async with self._restart_locks[index]:
            if self._num_restarts[index] != seen_restarts:
                return
            self._handles.pop(index, None)
            cell_id = reward_cell_id(self.name, cell_index=index)
            operations = self._capability.cell_operations()
            # A dead cell the liveness scan has not reached yet still holds its actors, and resume skips such a cell.
            await operations.suspend(cell_id=cell_id)
            await operations.resume(cell_id=cell_id)
            self._num_restarts[index] += 1
