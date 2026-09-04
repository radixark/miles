from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_MAX_DELAY_SECONDS", 0.001)


class _FakeCell:
    def __init__(self, *, ready: bool = False, needs_offload: bool = True):
        self.ready = ready
        self.meta = SimpleNamespace(needs_offload=needs_offload)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return self.ready


def _make_server(*, colocate: bool, expected_num_cells: int, cells: dict | None = None) -> RolloutServer:
    return RolloutServer(
        server_cells=cells if cells is not None else {},
        args=SimpleNamespace(colocate=colocate),
        context_lock=ContextLock("InferenceController"),
        expected_num_cells=expected_num_cells,
    )


class TestWaitExpectedNumCellsWhenColocated:
    async def test_cells_only_have_to_appear(self):
        """Colocated engines cannot load until the first weight update window, so readiness cannot be required."""
        srv = _make_server(colocate=True, expected_num_cells=2, cells={"a": _FakeCell(), "b": _FakeCell()})

        await asyncio.wait_for(srv.wait_expected_num_cells(), timeout=1)

    async def test_it_waits_while_cells_are_still_missing(self):
        """Starting a rollout with half the pool would run the first step on far too few engines."""
        cells: dict = {"a": _FakeCell()}
        srv = _make_server(colocate=True, expected_num_cells=2, cells=cells)

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)
        assert not task.done()

        cells["b"] = _FakeCell()
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsWhenDisaggregated:
    async def test_appearing_is_not_enough_the_engines_must_be_up(self):
        """A cell that has not finished loading yet cannot serve the first rollout."""
        srv = _make_server(colocate=False, expected_num_cells=1, cells={"a": _FakeCell(ready=False)})

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)

        assert not task.done()
        task.cancel()

    async def test_it_returns_once_every_engine_is_up(self):
        """This is the startup barrier that replaced the blocking wait inside cell startup."""
        cell = _FakeCell(ready=False)
        srv = _make_server(colocate=False, expected_num_cells=1, cells={"a": cell})

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)
        cell.ready = True
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsWithADedicatedEvalFleet:
    async def test_a_cell_outside_the_trainer_gpus_still_has_to_come_up(self):
        """A colocated run whose eval cells count as ready on arrival snapshots api clients that have no address yet."""
        srv = _make_server(
            colocate=True, expected_num_cells=1, cells={"eval": _FakeCell(ready=False, needs_offload=False)}
        )

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)

        assert not task.done()
        task.cancel()

    async def test_it_returns_once_the_eval_cell_is_up(self):
        """The barrier is what makes the eval fleet see engines that are actually addressable."""
        cell = _FakeCell(ready=False, needs_offload=False)
        srv = _make_server(colocate=True, expected_num_cells=1, cells={"eval": cell})

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)
        cell.ready = True

        await asyncio.wait_for(task, timeout=5)

    async def test_a_mixed_pool_waits_for_the_eval_cell_but_not_for_the_deferred_ones(self):
        """One colocated run holds both kinds of cell, so the wait cannot be decided per run."""
        eval_cell = _FakeCell(ready=False, needs_offload=False)
        srv = _make_server(
            colocate=True,
            expected_num_cells=2,
            cells={"shared": _FakeCell(ready=False, needs_offload=True), "eval": eval_cell},
        )

        task = asyncio.create_task(srv.wait_expected_num_cells())
        await asyncio.sleep(0)
        assert not task.done()

        eval_cell.ready = True
        await asyncio.wait_for(task, timeout=5)


class TestWaitExpectedNumCellsEdges:
    async def test_a_model_without_cells_does_not_wait(self):
        """A server expecting nothing must not hold up startup."""
        srv = _make_server(colocate=False, expected_num_cells=0)

        await asyncio.wait_for(srv.wait_expected_num_cells(), timeout=1)

    async def test_more_cells_than_expected_do_not_hang_the_wait(self):
        """An exact-match check would stall forever the moment the pool is bigger than planned."""
        srv = _make_server(
            colocate=True, expected_num_cells=1, cells={"a": _FakeCell(), "b": _FakeCell(), "c": _FakeCell()}
        )

        await asyncio.wait_for(srv.wait_expected_num_cells(), timeout=1)

    async def test_it_gives_up_instead_of_waiting_forever(self):
        """A pool that never comes up must surface as a failure rather than a silent hang."""
        srv = _make_server(colocate=True, expected_num_cells=1)

        with pytest.raises(Exception, match="Only 0/1 cells"):
            await srv.wait_expected_num_cells(timeout=0)
