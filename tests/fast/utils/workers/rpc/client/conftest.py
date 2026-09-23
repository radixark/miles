from __future__ import annotations

import pytest

from miles.utils.workers.rpc.client import call as rpc_call_module


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(rpc_call_module.asyncio, "sleep", fake_sleep)
    return sleeps
