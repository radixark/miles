from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from miles.utils.workers.reconcile.source_event import ReplaceEvent, SourceEvent


async def settle(iterations: int = 200) -> None:
    for _ in range(iterations):
        await asyncio.sleep(0)


class StreamEnd:
    pass


class StreamError:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class FakeSource:
    def __init__(self, *, fail_opens: int = 0, fail_calls: int = 0) -> None:
        self.open_count = 0
        self.closed_count = 0
        self._fail_opens = fail_opens
        self._fail_calls = fail_calls
        self._queues: list[asyncio.Queue[Any]] = []

    def __call__(self) -> AsyncGenerator[SourceEvent, None]:
        self.open_count += 1
        if self.open_count <= self._fail_calls:
            raise RuntimeError("fake source factory failure")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._queues.append(queue)
        return self._iterate(queue, fail=self.open_count <= self._fail_opens)

    async def _iterate(self, queue: asyncio.Queue[Any], *, fail: bool) -> AsyncGenerator[SourceEvent, None]:
        try:
            if fail:
                raise RuntimeError("fake source open failure")
            while True:
                item = await queue.get()
                if isinstance(item, StreamEnd):
                    return
                if isinstance(item, StreamError):
                    raise item.error
                yield item
        finally:
            self.closed_count += 1

    def emit(self, *events: Any) -> None:
        for event in events:
            self._queues[-1].put_nowait(event)


def make_pod(name: str, *, cell: str = "cell-a", resource_version: str = "1") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, resource_version=resource_version, labels={"cell": cell})
    )


def replace_of(*pods: Any) -> ReplaceEvent:
    return ReplaceEvent(objects={pod.metadata.name: pod for pod in pods})


def pod_cell(pod: Any) -> str:
    cell = pod.metadata.labels["cell"]
    assert isinstance(cell, str), f"pod has no usable cell label {pod=}"
    return cell
