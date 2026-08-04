from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from miles.utils.workers.reconcile.k8s_api import PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.source_event import ParentKey, ReplaceEvent, SourceEvent


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


class ParentKeyRecorder:
    def __init__(self) -> None:
        self.parent_keys: list[ParentKey] = []

    async def __call__(self, parent_key: ParentKey) -> None:
        self.parent_keys.append(parent_key)


class EventCollector:
    def __init__(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        self.events: list[SourceEvent] = []
        self.error: BaseException | None = None
        self._stream = stream
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async for event in self._stream:
                self.events.append(event)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.error = error

    async def close(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        await self._stream.aclose()
        assert self.error is None, f"stream failed with {self.error=}"


def make_pod(name: str, *, cell: str = "cell-a", resource_version: str = "1") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, resource_version=resource_version, labels={"cell": cell})
    )


def replace_of(*pods: Any) -> ReplaceEvent:
    return ReplaceEvent(objects={pod.metadata.name: pod for pod in pods})


def make_pod_list(pods: list[Any], *, resource_version: str) -> PodListPage:
    return PodListPage(pods=pods, resource_version=resource_version)


def pod_cell(pod: Any) -> str:
    cell = pod.metadata.labels["cell"]
    assert isinstance(cell, str), f"pod has no usable cell label {pod=}"
    return cell


class FakePodApi:
    def __init__(self) -> None:
        self.list_pages: deque[Any] = deque()
        self.stream_scripts: deque[list[Any]] = deque()
        self.list_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.closed_streams = 0
        self.aclosed_streams = 0

    async def list_pods(self, *, namespace: str, label_selector: str) -> Any:
        self.list_calls.append(dict(namespace=namespace, label_selector=label_selector))
        page = self.list_pages.popleft() if self.list_pages else None
        if isinstance(page, BaseException):
            raise page
        assert page is not None, "FakePodApi ran out of scripted list pages"
        return page

    async def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        self.stream_calls.append(
            dict(
                namespace=namespace,
                label_selector=label_selector,
                resource_version=resource_version,
                timeout_seconds=timeout_seconds,
            )
        )
        script = self.stream_scripts.popleft() if self.stream_scripts else None
        try:
            for item in script or []:
                if isinstance(item, BaseException):
                    raise item
                yield item
            if script is None:
                await asyncio.Event().wait()
        except GeneratorExit:
            self.aclosed_streams += 1
            raise
        finally:
            self.closed_streams += 1


class FakeApiException(Exception):
    def __init__(self, status: int | str) -> None:
        super().__init__(f"api exception {status}")
        self.status = status
