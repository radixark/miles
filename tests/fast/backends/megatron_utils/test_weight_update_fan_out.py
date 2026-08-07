import threading

import pytest

from miles.backends.megatron_utils.update_weight.common import begin_weight_update, end_weight_update


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
        """The requests are fired together, so a later engine is already in flight when an
        earlier one fails."""
        calls: list[str] = []
        gate = threading.Event()
        gate.set()
        clients = [_FailingClient(calls, 0), _FailingClient(calls, 1, gate=gate)]

        with pytest.raises(RuntimeError, match="boom"):
            begin_weight_update(clients)

        assert calls == ["slow-1"]
