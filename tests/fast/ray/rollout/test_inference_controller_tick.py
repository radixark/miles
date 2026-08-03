from __future__ import annotations

import asyncio

from types import SimpleNamespace

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.misc import SimpleTicker


class _RecordingCell:
    def __init__(self, *, error: Exception | None = None, delay: float = 0.0, cell_id: str = "cell"):
        self.tick_count = 0
        self.finished_count = 0
        self.meta = SimpleNamespace(cell_id=cell_id)
        self._error = error
        self._delay = delay

    async def tick(self) -> None:
        self.tick_count += 1
        if self._error is not None:
            raise self._error
        if self._delay:
            await asyncio.sleep(self._delay)
        self.finished_count += 1


class _StubServer:
    def __init__(self, server_cells: dict):
        self.server_cells = server_cells


def _make_controller(servers: dict) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.servers = servers
    controller._watcher_disposers = []
    controller._ticker = None
    return controller


def _start_ticker(controller: InferenceController) -> None:
    controller._ticker = SimpleTicker(controller._tick_cells, interval_seconds=0.0)


class TestTickCells:
    async def test_it_drives_every_cell_of_every_server(self):
        """A cell only makes progress when ticked, so no server may be left out of the sweep."""
        first, second, third = _RecordingCell(), _RecordingCell(), _RecordingCell()
        controller = _make_controller(
            {"default": _StubServer({"a": first, "b": second}), "frozen": _StubServer({"c": third})}
        )

        await controller._tick_cells()

        assert [cell.tick_count for cell in (first, second, third)] == [1, 1, 1]

    async def test_one_failing_cell_does_not_let_its_siblings_escape_the_sweep(self):
        """A sweep that returns early would release the lock while sibling ticks still mutate state."""
        broken = _RecordingCell(error=RuntimeError("cell exploded"), cell_id="broken")
        slow = _RecordingCell(delay=0.02, cell_id="slow")
        controller = _make_controller({"default": _StubServer({"a": broken, "b": slow})})

        await controller._tick_cells()

        assert slow.finished_count == 1

    async def test_a_wedged_cell_cannot_stall_the_sweep_forever(self, monkeypatch):
        """A hung engine holds the controller lock for as long as its tick runs, so it must be bounded."""
        wedged = _RecordingCell(delay=60.0, cell_id="wedged")
        healthy = _RecordingCell(cell_id="healthy")
        controller = _make_controller({"default": _StubServer({"a": wedged, "b": healthy})})
        monkeypatch.setattr(inference_controller_module, "CELL_TICK_TIMEOUT_SECONDS", 0.01)

        await controller._tick_cells()

        assert wedged.finished_count == 0
        assert healthy.finished_count == 1

    async def test_a_cell_added_after_the_loop_started_is_picked_up(self):
        """Cells appear from reconcile long after startup, so the sweep must re-read the bookkeeping."""
        srv = _StubServer({})
        controller = _make_controller({"default": srv})

        _start_ticker(controller)
        await asyncio.sleep(0.01)
        late = _RecordingCell()
        srv.server_cells["late"] = late
        await asyncio.sleep(0.02)
        await controller.dispose()

        assert late.tick_count > 0

    async def test_the_sweep_keeps_running_after_one_cell_raises(self):
        """One wedged engine must not stop every other cell from making progress."""
        broken, healthy = _RecordingCell(error=RuntimeError("cell exploded")), _RecordingCell()
        controller = _make_controller({"default": _StubServer({"a": broken, "b": healthy})})

        _start_ticker(controller)
        await asyncio.sleep(0.02)
        await controller.dispose()

        assert broken.tick_count > 1
        assert healthy.tick_count > 1


class TestControllerDisposal:
    async def test_dispose_stops_the_ticker(self):
        """A surviving loop would keep dialing engines after the controller is gone."""
        cell = _RecordingCell()
        controller = _make_controller({"default": _StubServer({"a": cell})})

        _start_ticker(controller)
        await asyncio.sleep(0.02)
        await controller.dispose()
        ticks_after_dispose = cell.tick_count
        await asyncio.sleep(0.02)

        assert cell.tick_count == ticks_after_dispose

    async def test_dispose_without_a_running_ticker_is_harmless(self):
        """debug_train_only never starts the ticker, and teardown still has to work."""
        controller = _make_controller({})

        await controller.dispose()

        assert controller._ticker is None
