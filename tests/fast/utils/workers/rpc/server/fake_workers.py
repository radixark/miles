from __future__ import annotations

import asyncio
from typing import NamedTuple

from miles.utils.workers.rpc.common.metadata import RpcMethodSpec, rpc
from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


class SyncAndAsyncWorker:
    def demo_sync(self, value: int) -> int:
        return value + 1

    def demo_sync_raises(self) -> None:
        raise RuntimeError("boom")

    @rpc(concurrency_group="left")
    def demo_on_left(self) -> str:
        return "left"

    @rpc(concurrency_group="left")
    def demo_also_on_left(self) -> str:
        return "also-left"

    async def demo_async(self) -> str:
        return "async"


class AsyncOnlyWorker:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def demo_async(self) -> str:
        return "async"

    async def demo_async_raises(self) -> None:
        raise RuntimeError("boom")

    async def demo_async_blocks(self) -> None:
        self.started.set()
        await asyncio.Event().wait()


class OutcomeRecorder:
    def __init__(self) -> None:
        self.outcomes: list[CallStatusResponse] = []

    def finish(self, *, outcome: CallStatusResponse) -> None:
        self.outcomes.append(outcome)


class ExecutorUnderTest(NamedTuple):
    executor: RpcCallExecutor
    specs: dict[str, RpcMethodSpec]
