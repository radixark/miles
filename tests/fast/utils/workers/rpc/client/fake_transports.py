from __future__ import annotations

import asyncio
from typing import Any, NamedTuple

import httpx

_CALL_STATUS_FRAGMENT = "/v1/calls/"


def _is_poll(request: httpx.Request) -> bool:
    return request.method == "GET" and _CALL_STATUS_FRAGMENT in str(request.url)


class StallingPollTransport(httpx.AsyncBaseTransport):
    def __init__(self, app: Any, *, stalled_polls: int, stall_seconds: float = 60.0) -> None:
        self.polls = 0
        self._inner = httpx.ASGITransport(app=app)
        self._stalled_polls = stalled_polls
        self._stall_seconds = stall_seconds

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if _is_poll(request):
            self.polls += 1
            if self.polls <= self._stalled_polls:
                await asyncio.sleep(self._stall_seconds)
                raise AssertionError("the stalled poll should have been abandoned")
        return await self._inner.handle_async_request(request)


class PollWindow(NamedTuple):
    client_seconds: float
    server_seconds: float


class PollWindowRecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, app: Any) -> None:
        self.poll_windows: list[PollWindow] = []
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if _is_poll(request):
            self.poll_windows.append(
                PollWindow(
                    client_seconds=float(request.extensions["timeout"]["read"]),
                    server_seconds=float(request.url.params["timeout"]),
                )
            )
        return await self._inner.handle_async_request(request)
