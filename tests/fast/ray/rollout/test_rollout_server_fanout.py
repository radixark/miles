from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import NamedHostAndPorts


class _RecordingCell:
    def __init__(self, *, cell_id: str, needs_offload: bool, addressable: bool = True):
        self.meta = SimpleNamespace(needs_offload=needs_offload, cell_id=cell_id, num_gpus_per_engine=1, gpu_offset=0)
        self.is_pending_weights_or_serving = addressable
        self.is_faulted = False
        self.api_client = f"client-{cell_id}"
        self.calls: list[tuple[str, dict]] = []

    def mark_faulted(self) -> None:
        self.is_faulted = True

    async def offload(self, tags):
        self.calls.append(("offload", dict(tags=tags)))
        return f"offloaded-{self.meta.cell_id}"

    async def onload(self, tags):
        self.calls.append(("onload", dict(tags=tags)))
        return f"onloaded-{self.meta.cell_id}"

    async def check_weights(self, action, allow_quant_error, selector, skip_list):
        self.calls.append(
            (
                "check_weights",
                dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list),
            )
        )
        return f"checked-{self.meta.cell_id}"

    async def abort_all(self):
        self.calls.append(("abort_all", {}))


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"fanning out to the cells must not resolve {worker_name}")


def _make_server(cells: list[_RecordingCell], **overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={cell.meta.cell_id: cell for cell in cells},
        args=make_args(colocate=True),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


class TestMemoryFanOut:
    async def test_only_the_cells_sharing_gpus_with_the_trainer_give_memory_back(self):
        """A resident engine told to release would drop the weights nobody is going to reload."""
        colocated = _RecordingCell(cell_id="a", needs_offload=True)
        resident = _RecordingCell(cell_id="b", needs_offload=False)
        srv = _make_server([colocated, resident])

        async with srv.context_lock:
            await srv.offload()

        assert [name for name, _ in colocated.calls] == ["offload"]
        assert resident.calls == []

    @pytest.mark.parametrize("op", ["offload", "onload"])
    async def test_the_requested_tags_reach_every_cell_unchanged(self, op):
        """Resuming the wrong tag set brings back a different slice of the engine's memory."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=True) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await getattr(srv, op)(tags=["weights"])

        assert results == [f"{op}ed-{i}" for i in range(3)]
        assert all(cell.calls == [(op, dict(tags=["weights"]))] for cell in cells)

    async def test_a_cell_without_an_address_yet_is_left_alone(self):
        """Reconcile can add a gated cell mid-window; dialling it asserts inside the cell and
        takes down a weight update the cell was never part of."""
        gated = _RecordingCell(cell_id="gated", needs_offload=True, addressable=False)
        serving = _RecordingCell(cell_id="serving", needs_offload=True)
        srv = _make_server([gated, serving])

        async with srv.context_lock:
            await srv.offload()

        assert gated.calls == []
        assert [name for name, _ in serving.calls] == ["offload"]


class TestCheckWeightsFanOut:
    async def test_every_addressable_cell_is_checked_with_the_same_arguments(self):
        """Narrowing this to one cell would verify one engine of N and call them all equal."""
        cells = [_RecordingCell(cell_id=str(i), needs_offload=False) for i in range(3)]
        srv = _make_server(cells)

        async with srv.context_lock:
            results = await srv.check_weights(
                action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]
            )

        assert results == [f"checked-{i}" for i in range(3)]
        assert all(
            cell.calls
            == [
                (
                    "check_weights",
                    dict(action="snapshot", allow_quant_error=True, selector="lora", skip_list=["x"]),
                )
            ]
            for cell in cells
        )

    async def test_a_cell_without_an_address_yet_is_not_checked(self):
        """The check runs during the weight update window, which a gated cell has not entered."""
        gated = _RecordingCell(cell_id="gated", needs_offload=False, addressable=False)
        srv = _make_server([gated])

        async with srv.context_lock:
            assert await srv.check_weights(action="snapshot") == []

        assert gated.calls == []


class _NeverAnsweringCell(_RecordingCell):
    async def abort_all(self):
        self.calls.append(("abort_all", {}))
        await asyncio.sleep(3600)


class TestAbortFanOut:
    async def test_every_addressable_cell_is_asked_to_abort(self):
        """A take-over aborts what the previous script left generating, on every engine it can still reach."""
        cells = [_RecordingCell(cell_id="a", needs_offload=False), _RecordingCell(cell_id="b", needs_offload=False)]
        srv = _make_server(cells)

        async with srv.context_lock:
            await srv.abort_all()

        assert all(cell.calls == [("abort_all", {})] for cell in cells)

    async def test_a_cell_that_never_answers_is_given_up_on(self, monkeypatch):
        """The abort runs under the controller's lock, so an engine that never answers would wedge every caller."""
        monkeypatch.setattr(rollout_server_module, "ABORT_ALL_TIMEOUT_SECONDS", 0.01)
        cells = [
            _RecordingCell(cell_id="a", needs_offload=False),
            _NeverAnsweringCell(cell_id="b", needs_offload=False),
        ]
        srv = _make_server(cells)

        async with srv.context_lock:
            with pytest.raises(TimeoutError):
                await srv.abort_all()


class _DeadCell(_RecordingCell):
    async def offload(self, tags):
        self.calls.append(("offload", dict(tags=tags)))
        raise TimeoutError(f"Timeout while flushing cache of {self.meta.cell_id}")

    async def onload(self, tags):
        self.calls.append(("onload", dict(tags=tags)))
        raise TimeoutError(f"Timeout while waking {self.meta.cell_id}")


class TestMemoryFanOutSurvivesADeadEngine:
    async def test_a_killed_engine_does_not_fail_the_offload_of_the_cells_that_are_still_up(self):
        """A cell that was just killed cannot release memory, and blaming the driver for that ends the whole run."""
        dead = _DeadCell(cell_id="dead", needs_offload=True)
        alive = _RecordingCell(cell_id="alive", needs_offload=True)
        server = _make_server([dead, alive])

        async with server.context_lock:
            results = await server.offload(tags=["weights"])

        assert results == ["offloaded-alive"]
        assert alive.calls == [("offload", dict(tags=["weights"]))]

    async def test_a_killed_engine_does_not_fail_the_onload_of_the_cells_that_are_still_up(self):
        """The same holds on the way back: one dead engine must not stop the survivors from waking."""
        dead = _DeadCell(cell_id="dead", needs_offload=True)
        alive = _RecordingCell(cell_id="alive", needs_offload=True)
        server = _make_server([dead, alive])

        async with server.context_lock:
            results = await server.onload(tags=["weights"])

        assert results == ["onloaded-alive"]

    async def test_an_offload_that_reaches_no_engine_at_all_still_raises(self):
        """Silently swallowing a fleet-wide failure would leave the trainer without the GPUs it is owed."""
        server = _make_server([_DeadCell(cell_id="dead", needs_offload=True)])

        with pytest.raises(TimeoutError):
            async with server.context_lock:
                await server.offload(tags=["weights"])


class _WedgedCell(_RecordingCell):
    async def offload(self, tags):
        self.calls.append(("offload", dict(tags=tags)))
        await asyncio.get_running_loop().create_future()


class TestMemoryFanOutBoundsAWedgedEngine:
    async def test_an_engine_that_accepted_the_connection_and_never_answered_does_not_hold_the_fleet(
        self, monkeypatch
    ):
        """The http client reads without a timeout, so a wedged engine would hold the lock and the health pause forever."""
        monkeypatch.setattr(rollout_server_module, "MEMORY_MOVE_TIMEOUT_SECONDS", 0.05)
        wedged = _WedgedCell(cell_id="wedged", needs_offload=True)
        alive = _RecordingCell(cell_id="alive", needs_offload=True)
        server = _make_server([wedged, alive])

        async with server.context_lock:
            results = await server.offload(tags=["weights"])

        assert results == ["offloaded-alive"]


class TestWeightUpdateSnapshotSkipsAFaultedCell:
    def _server(self) -> tuple[RolloutServer, list[_RecordingCell]]:
        cells = [_RecordingCell(cell_id=name, needs_offload=False) for name in ("a", "b", "c")]
        for offset, cell in enumerate(cells):
            cell.meta.gpu_offset = offset
        return _make_server(cells), cells

    async def test_a_faulted_cell_leaves_the_engine_lists_before_the_next_broadcast(self):
        """A killed engine left in the snapshot makes the trainer open an NCCL group nobody completes."""
        server, cells = self._server()

        async with server.context_lock:
            server.mark_cell_faulted("b")

            assert server.api_clients == ["client-a", "client-c"]
            assert server.engine_gpu_offsets == [0, 2]
            assert server.engine_gpu_counts == [1, 1]

    async def test_a_faulted_cell_is_no_longer_addressable(self):
        """The last-replica guard and the memory fan-out read the same view, so one mark serves both."""
        server, cells = self._server()

        async with server.context_lock:
            server.mark_cell_faulted("b")

            assert server.addressable_cell_ids() == ["a", "c"]
