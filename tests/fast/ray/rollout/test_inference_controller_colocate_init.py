from __future__ import annotations

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController


class _FakeCell:
    def __init__(self, *, state: str = "uninitialized", init_gate: asyncio.Event | None = None):
        self.state = state
        self.init_count = 0
        self.init_started = asyncio.Event()
        self._init_gate = init_gate

    async def init(self) -> None:
        self.init_count += 1
        self.init_started.set()
        if self._init_gate is not None:
            await self._init_gate.wait()
        if self.state == "uninitialized":
            self.state = "initializing"

    def become_ready(self) -> None:
        self.state = "pending_weights"

    @property
    def is_uninitialized(self) -> bool:
        return self.state == "uninitialized"

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return self.state == "pending_weights"


class _StubServer:
    def __init__(self, server_cells: dict):
        self.server_cells = server_cells


def _make_controller(servers: dict, *, colocate: bool) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = make_args(colocate=colocate)
    controller.servers = servers
    return controller


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(inference_controller_module, "CELLS_READY_POLL_INTERVAL_SECONDS", 0.0)


class TestEnsureCellsReady:
    async def test_a_disaggregated_cell_is_never_initialized_by_the_window(self):
        """Disaggregated cells initialize on arrival, so the window only has to wait for them."""
        cell = _FakeCell(state="pending_weights")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        await asyncio.wait_for(controller._ensure_cells_ready(), timeout=1)

        assert cell.init_count == 0

    async def test_a_disaggregated_engine_still_loading_is_waited_for(self):
        """Weights pushed into an engine that is not up yet would be lost."""
        cell = _FakeCell(state="initializing")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        task = asyncio.create_task(controller._ensure_cells_ready())
        await asyncio.sleep(0.02)
        assert not task.done()
        cell.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert cell.init_count == 0

    async def test_it_initializes_every_colocated_cell_that_is_still_waiting(self):
        """This window is the only moment a colocated engine may claim gpu memory."""
        first, second = _FakeCell(), _FakeCell()
        controller = _make_controller(
            {"default": _StubServer({"a": first}), "frozen": _StubServer({"b": second})}, colocate=True
        )

        task = asyncio.create_task(controller._ensure_cells_ready())
        await asyncio.sleep(0)
        first.become_ready()
        second.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert (first.init_count, second.init_count) == (1, 1)

    async def test_an_already_running_cell_is_not_initialized_again(self):
        """Re-initializing a live engine would tear down the memory it already holds."""
        running = _FakeCell(state="pending_weights")
        controller = _make_controller({"default": _StubServer({"a": running})}, colocate=True)

        await asyncio.wait_for(controller._ensure_cells_ready(), timeout=1)

        assert running.init_count == 0

    async def test_it_waits_until_the_engines_finished_loading(self):
        """Returning early would let the trainer push weights into an engine that is not up."""
        cell = _FakeCell()
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=True)

        task = asyncio.create_task(controller._ensure_cells_ready())
        await asyncio.sleep(0.02)
        assert not task.done()

        cell.become_ready()
        await asyncio.wait_for(task, timeout=1)

    async def test_a_cell_arriving_mid_wait_is_initialized_in_the_same_window(self):
        """The window's contract is that every cell that exists is ready when it returns."""
        early = _FakeCell()
        srv = _StubServer({"a": early})
        controller = _make_controller({"default": srv}, colocate=True)

        task = asyncio.create_task(controller._ensure_cells_ready())
        await asyncio.sleep(0)
        late = _FakeCell()
        srv.server_cells["late"] = late
        early.become_ready()
        await asyncio.wait_for(late.init_started.wait(), timeout=1)
        late.become_ready()
        await asyncio.wait_for(task, timeout=1)

        assert late.init_count == 1

    async def test_a_run_without_any_cells_yet_returns_immediately(self):
        """The first window can legitimately find nothing to do."""
        controller = _make_controller({"default": _StubServer({})}, colocate=True)

        await asyncio.wait_for(controller._ensure_cells_ready(), timeout=1)

    async def test_the_cells_are_initialized_concurrently(self):
        """Activating a gate blocks until the engine reaches it, so serialising would add minutes per cell."""
        gate = asyncio.Event()
        first, second = _FakeCell(init_gate=gate), _FakeCell(init_gate=gate)
        controller = _make_controller({"default": _StubServer({"a": first, "b": second})}, colocate=True)

        task = asyncio.create_task(controller._ensure_cells_ready())
        await asyncio.wait_for(asyncio.gather(first.init_started.wait(), second.init_started.wait()), timeout=1.0)
        gate.set()
        for cell in (first, second):
            cell.become_ready()
        await asyncio.wait_for(task, timeout=1.0)

        assert (first.init_count, second.init_count) == (1, 1)

    async def test_an_engine_that_never_comes_up_eventually_times_out(self, monkeypatch):
        """A silent hang here would stall training with no explanation."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0.0)
        cell = _FakeCell(state="initializing")
        controller = _make_controller({"default": _StubServer({"a": cell})}, colocate=False)

        with pytest.raises(TimeoutError):
            await controller._ensure_cells_ready()
