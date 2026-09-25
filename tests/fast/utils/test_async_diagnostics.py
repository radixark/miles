import asyncio
import contextvars
import gc
import json
import logging
import os
import subprocess
import sys

import pytest

from miles.utils.async_diagnostics import async_diagnostic_scope, configure_async_diagnostics
from miles.utils.logging_utils import configure_strict_async_warnings


@pytest.fixture
def diagnostics(monkeypatch):
    monkeypatch.setenv("MILES_ASYNC_DIAGNOSTICS", "1")
    depth = sys.get_coroutine_origin_tracking_depth()
    yield
    sys.set_coroutine_origin_tracking_depth(depth)


def events(caplog):
    return [
        json.loads(r.message.split("async_diagnostic ", 1)[1])
        for r in caplog.records
        if "async_diagnostic " in r.message
    ]


def test_cancel_and_timeout_provenance(diagnostics, caplog):
    async def trial():
        with async_diagnostic_scope("trial-123", timeout_s=0.01):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.sleep(60), timeout=0.01)
            task = asyncio.create_task(asyncio.sleep(60), context=contextvars.copy_context())
            assert task.cancel("SECRET cancellation message")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not task.cancel()

    async def run():
        configure_async_diagnostics(asyncio.get_running_loop())
        # Run the trial in a traced task: Python 3.12's wait_for cancels the
        # awaiting task itself instead of wrapping the coroutine in a new one.
        await asyncio.create_task(trial())

    asyncio.run(run())
    cancels = [e for e in events(caplog) if e["event"] == "task_cancel"]
    assert len(cancels) == 3
    assert all(e["scope"] == "trial-123" for e in cancels)
    assert any(any(f[2] in {"_cancel_and_wait", "_on_timeout"} for f in e["caller_stack"]) for e in cancels)
    assert "SECRET" not in caplog.text


def test_disabled_and_existing_factory(monkeypatch, caplog):
    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.delenv("MILES_ASYNC_DIAGNOSTICS", raising=False)
        configure_async_diagnostics(loop)
        assert loop.get_task_factory() is None

        def factory(loop, coro, **kw):
            return asyncio.Task(coro, loop=loop, **kw)

        loop.set_task_factory(factory)
        monkeypatch.setenv("MILES_ASYNC_DIAGNOSTICS", "1")
        configure_async_diagnostics(loop)
        assert loop.get_task_factory() is factory

    asyncio.run(run())
    assert any(e["event"] == "task_factory_conflict" for e in events(caplog))


def test_current_task_timeout_and_scope_reset(diagnostics, caplog):
    async def trial():
        with async_diagnostic_scope("outer-trial", timeout_s=1):
            with async_diagnostic_scope("inner-trial", timeout_s=0.01):
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.01):
                        await asyncio.sleep(60)
            assert asyncio.current_task()._diagnostic_scope == "outer-trial"
        assert asyncio.current_task()._diagnostic_scope is None

    async def run():
        configure_async_diagnostics(asyncio.get_running_loop())
        await asyncio.create_task(trial())

    asyncio.run(run())
    cancels = [e for e in events(caplog) if e["event"] == "task_cancel"]
    assert len(cancels) == 1
    assert cancels[0]["scope"] == "inner-trial"


@pytest.mark.parametrize("policy,returncode", [("error", 1), ("log", 0)])
def test_executor_cancellation_survival(policy, returncode):
    # Reproduce a late coroutine result from a cancelled executor future without
    # a provider dependency or network calls. A separate process tests os._exit.
    code = """
import asyncio, gc, threading
from miles.utils.logging_utils import configure_strict_async_warnings
configure_strict_async_warnings()
entered = threading.Event()
release = threading.Event()
async def request(): pass
def prepare():
    value = request()
    entered.set()
    release.wait(10)
    return value
async def run():
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, prepare)
    while not entered.is_set():
        await asyncio.sleep(.001)
    future.cancel()
    release.set()
    await loop.shutdown_default_executor()
    await asyncio.sleep(0)
    gc.collect()
asyncio.run(run())
print("survived")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MILES_ASYNC_WARNING_POLICY": policy},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == returncode, result.stderr
    assert '"event": "unawaited_coroutine"' in result.stderr
    assert ("survived" in result.stdout) == (policy == "log")


def test_invalid_policy(monkeypatch):
    monkeypatch.setenv("MILES_ASYNC_WARNING_POLICY", "typo")
    with pytest.raises(ValueError, match="MILES_ASYNC_WARNING_POLICY"):
        configure_strict_async_warnings()


def test_lifecycle_events_are_informational_and_defects_are_warnings(diagnostics, caplog):
    caplog.set_level(logging.INFO, logger="miles.utils.async_diagnostics")

    async def run():
        configure_async_diagnostics(asyncio.get_running_loop())
        with async_diagnostic_scope("trial-ok", timeout_s=1):
            await asyncio.sleep(0)
        task = asyncio.create_task(asyncio.sleep(60))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    levels = {
        r.message.split('"event": "', 1)[1].split('"', 1)[0]: r.levelno
        for r in caplog.records
        if "async_diagnostic " in r.message
    }
    assert levels["enabled"] == levels["scope_start"] == levels["scope_end"] == logging.INFO
    assert levels["task_cancel"] == logging.WARNING


def test_task_ids_are_process_unique_counters(diagnostics, caplog):
    async def run():
        configure_async_diagnostics(asyncio.get_running_loop())
        for _ in range(3):
            task = asyncio.create_task(asyncio.sleep(60))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            del task
            gc.collect()

    asyncio.run(run())
    cancels = [e for e in events(caplog) if e["event"] == "task_cancel"]
    ids = [e["task_id"] for e in cancels]
    assert len(ids) == 3 and len(set(ids)) == 3 and ids == sorted(ids)
    assert all(e["caller_task_id"] not in ids for e in cancels)
    assert all(isinstance(f[0], str) and isinstance(f[2], str) for e in cancels for f in e["caller_stack"])


def test_cancel_skips_diagnostic_work_when_warnings_are_disabled(diagnostics, caplog, monkeypatch):
    calls = []
    monkeypatch.setattr("miles.utils.async_diagnostics._caller_stack", lambda frame: calls.append(frame) or [])
    logging.getLogger("miles.utils.async_diagnostics").setLevel(logging.ERROR)
    try:

        async def run():
            configure_async_diagnostics(asyncio.get_running_loop())
            task = asyncio.create_task(asyncio.sleep(60))
            assert task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
    finally:
        logging.getLogger("miles.utils.async_diagnostics").setLevel(logging.NOTSET)
    assert calls == []
    assert not [e for e in events(caplog) if e["event"] == "task_cancel"]
