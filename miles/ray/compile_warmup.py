import asyncio

from miles.utils.async_utils import Disposer, eager_create_task
from miles.utils.data import RolloutDataPack, remove_rollout_data_refs
from miles.utils.workers.worker_handle import BaseWorkerHandle


class CompileWarmup:
    """The trainer compiles its first step's programs (--compile-warmup) while eval-before-train and rollout 0 run.

    Its ranks run one call at a time, so eval-before-train's weight export waits behind the warmup.
    """

    def __init__(self, task: asyncio.Task[None] | None) -> None:
        self._task = task

    @classmethod
    async def start(
        cls, args, *, actor_model: BaseWorkerHandle, rollout_executor: BaseWorkerHandle, disposer: Disposer
    ) -> "CompileWarmup":
        if not args.compile_warmup:
            return cls(task=None)
        packs = await rollout_executor.put_compile_warmup_data()
        warmup = cls(task=await eager_create_task(_warm_up(args, actor_model=actor_model, packs=packs)))
        disposer.add(warmup)
        return warmup

    async def wait(self) -> None:
        """Called before every train: the first one waits for the warmup, the rest return at once."""
        if self._task is not None:
            await self._task

    async def dispose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


async def _warm_up(args, *, actor_model: BaseWorkerHandle, packs: list[RolloutDataPack]) -> None:
    try:
        await actor_model.warm_up(rollout_data_packs=packs)
    finally:
        for pack in packs:
            remove_rollout_data_refs(args, pack)
