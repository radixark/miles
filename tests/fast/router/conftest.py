"""Shared fixtures for router tests.

The class-scoped ``router_env`` (mock SGLang upstream + session server) is
defined in test_sessions.py; re-exported here so additive test modules can
depend on it without importing a name their own test signatures would shadow.
"""

import asyncio
from collections.abc import Callable

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import router_env  # noqa: F401

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer


@pytest.fixture
def synchronize_mock_requests(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    def synchronize(parties: int) -> None:
        barriers: dict[MockSGLangServer, asyncio.Barrier] = {}

        async def handle(
            server: MockSGLangServer, request: Request, compute_fn: Callable[[dict], dict]
        ) -> JSONResponse:
            payload = await request.json()
            server.request_log.append(payload)
            with server._concurrency.track():
                if server not in barriers:
                    barriers[server] = asyncio.Barrier(parties)
                await asyncio.wait_for(barriers[server].wait(), timeout=30)
                response = compute_fn(payload)
            return JSONResponse(content=response)

        monkeypatch.setattr(MockSGLangServer, "_handle_generate_like_request", handle)

    return synchronize
