from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell, make_cell

from miles.ray.train.group import TrainerController

pytestmark = pytest.mark.asyncio


def _make_controller(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace()
    return group


def _reconcile_calls_of(cell) -> list:
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == "reconcile_adapters"]
        for handle in cell._get_actor_handles()
    ]


class TestReconcileAdapters:
    async def test_every_worker_of_every_cell_is_asked_to_reconcile(self):
        """Each rank of every cell owns its adapter registry, so all of them must be reached."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        group = _make_controller(cells)

        await group.reconcile_adapters()

        for cell in cells:
            assert _reconcile_calls_of(cell) == [[("reconcile_adapters", (), {})]] * 2

    async def test_a_failing_worker_in_a_later_cell_propagates_instead_of_being_swallowed(self):
        """A stale adapter set would corrupt routing, so a failure in any cell must reach the caller."""
        cells = [make_cell(index) for index in range(2)]
        ray.get(cells[1]._get_actor_handles()[0].set_fail_methods.remote(["reconcile_adapters"]))
        group = _make_controller(cells)

        with pytest.raises(Exception, match="Injected failure"):
            await group.reconcile_adapters()
