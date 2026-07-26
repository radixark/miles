import asyncio
import concurrent.futures
import threading
from unittest.mock import patch

import pytest

from miles.backends.megatron_utils.update_weight.common import begin_weight_update, end_weight_update
from miles.utils import async_utils

_COMMON_MODULE = "miles.backends.megatron_utils.update_weight.common"


class _RecordingClient:
    def __init__(self, calls: list[str], engine_index: int, selectors: list[str] | None = None):
        self._calls = calls
        self._engine_index = engine_index
        self._selectors = selectors if selectors is not None else []

    async def begin_weight_update(self, selector: str = "all"):
        self._calls.append(f"begin-{self._engine_index}")
        self._selectors.append(selector)
        return {"success": True}

    async def end_weight_update(self):
        self._calls.append(f"end-{self._engine_index}")
        return {"success": True}


class _FailingClient:
    def __init__(self, calls: list[str], engine_index: int, gate: threading.Event | None = None):
        self._calls = calls
        self._engine_index = engine_index
        self._gate = gate

    async def begin_weight_update(self, selector: str = "all"):
        if self._gate is not None:
            self._gate.wait(timeout=30)
            self._calls.append(f"slow-{self._engine_index}")
            return {"success": True}
        raise RuntimeError("boom")

    async def end_weight_update(self):
        return {"success": True}


class _EndFailingClient:
    def __init__(self, calls: list[str], engine_index: int):
        self._calls = calls
        self._engine_index = engine_index

    async def begin_weight_update(self, selector: str = "all"):
        self._calls.append(f"begin-{self._engine_index}")
        return {"success": True}

    async def end_weight_update(self):
        raise RuntimeError("close failed")


class _GatedEndClient:
    def __init__(self, calls: list[str], engine_index: int, gate: threading.Event):
        self._calls = calls
        self._engine_index = engine_index
        self._gate = gate

    async def begin_weight_update(self, selector: str = "all"):
        self._calls.append(f"begin-{self._engine_index}")
        return {"success": True}

    async def end_weight_update(self):
        if not await asyncio.to_thread(self._gate.wait, 5):
            raise TimeoutError("end_weight_update gate timed out")
        self._calls.append(f"end-{self._engine_index}")
        return {"success": True}


class _ObservedFuture:
    def __init__(self, future: concurrent.futures.Future, result_started: threading.Event):
        self._future = future
        self._result_started = result_started

    def result(self):
        self._result_started.set()
        return self._future.result()


class TestWeightUpdateSessionFanOut:
    """The session brackets must reach every engine, not just the first."""

    def test_begin_and_end_reach_every_engine(self):
        """A missed engine would load weights outside a session and corrupt them."""
        calls: list[str] = []
        clients = [_RecordingClient(calls, i) for i in range(4)]

        begin_weight_update(clients)
        end_weight_update(clients)

        assert sorted(calls[:4]) == ["begin-0", "begin-1", "begin-2", "begin-3"]
        assert sorted(calls[4:]) == ["end-0", "end-1", "end-2", "end-3"]

    def test_a_failing_engine_fails_the_session(self):
        """A silently swallowed failure would leave that engine loading outside a session."""
        calls: list[str] = []

        with pytest.raises(RuntimeError, match="boom"):
            begin_weight_update([_RecordingClient(calls, 0), _FailingClient(calls, 1)])

        assert calls == ["begin-0"]

    def test_every_engine_is_asked_before_the_failure_surfaces(self):
        """A later engine must already be in flight when an earlier engine fails."""
        calls: list[str] = []
        gate = threading.Event()
        gate.set()
        clients = [_FailingClient(calls, 0), _FailingClient(calls, 1, gate=gate)]

        with pytest.raises(RuntimeError, match="boom"):
            begin_weight_update(clients)

        assert calls == ["slow-1"]

    def test_end_failure_propagates_after_every_engine_has_settled(self):
        """A failure must surface only after every engine has finished closing its session."""
        calls: list[str] = []
        gate = threading.Event()
        result_started = threading.Event()
        clients = [_EndFailingClient(calls, 0), _GatedEndClient(calls, 1, gate)]
        original_submit = async_utils.submit
        submission_count = 0

        def observed_submit(coro):
            nonlocal submission_count
            future = original_submit(coro)
            submission_count += 1
            return _ObservedFuture(future, result_started) if submission_count == 2 else future

        with patch(f"{_COMMON_MODULE}.async_utils.submit", side_effect=observed_submit):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                outcome = executor.submit(end_weight_update, clients)
                try:
                    assert result_started.wait(timeout=5)
                    assert not outcome.done()
                    assert calls == []
                    gate.set()
                    with pytest.raises(RuntimeError, match="close failed"):
                        outcome.result(timeout=5)
                finally:
                    gate.set()

        assert submission_count == 2
        assert calls == ["end-1"]
