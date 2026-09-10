import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.utils.soak import action


@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_commands_are_reaped_before_the_error_escapes(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """Timeout and cancellation cannot leave the local command or its output task running."""
    started = asyncio.Event()
    killed = asyncio.Event()
    reaped = asyncio.Event()
    signals: list[tuple[int, int]] = []

    async def communicate() -> tuple[bytes, bytes]:
        started.set()
        await killed.wait()
        reaped.set()
        return b"", b""

    def kill(pid: int, signal: int) -> None:
        signals.append((pid, signal))
        killed.set()

    process = SimpleNamespace(pid=123, returncode=-9, communicate=communicate)
    monkeypatch.setattr(action.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(action.os, "killpg", kill)
    task = asyncio.create_task(action.run_command(["kubectl", "get", "pods"], timeout_seconds=3600 if cancel else 0))
    if cancel:
        await started.wait()
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert reaped.is_set()
    assert signals == [(123, action.signal.SIGKILL)]
