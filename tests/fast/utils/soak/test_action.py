import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.utils.soak import action


async def test_cancellation_during_spawn_waits_for_the_handle_then_kills_the_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during subprocess setup must not abandon an already spawning command."""
    spawning = asyncio.Event()
    release = asyncio.Event()
    killed = asyncio.Event()
    reaped = asyncio.Event()

    async def communicate() -> tuple[bytes, bytes]:
        await killed.wait()
        reaped.set()
        return b"", b""

    process = SimpleNamespace(pid=123, returncode=-9, communicate=communicate)

    async def create(*args: str, **kwargs: object) -> object:
        spawning.set()
        await release.wait()
        return process

    def kill(pid: int, signal: int) -> None:
        assert pid == 123 and signal == action.signal.SIGKILL
        killed.set()

    monkeypatch.setattr(action.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(action.os, "killpg", kill)
    task = asyncio.create_task(action.run_command(["launcher"], timeout_seconds=3600))
    await spawning.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed.is_set() and reaped.is_set()


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
