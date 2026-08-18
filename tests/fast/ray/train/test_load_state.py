from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell

from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train import group as group_module
from miles.ray.train.group import TrainerController
from miles.utils.init_once import InitOnce

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _expected_controller_fleet(monkeypatch):
    monkeypatch.setattr(group_module, "compute_trainer_num_cells", lambda args, *, role: args.expected_num_cells)


class _FakeCell:
    def __init__(self, *, cell_index: int, restored: list[int], is_alive: bool = True) -> None:
        self.cell_index = cell_index
        self.restored = restored
        self.is_alive = is_alive
        self.cell_id = f"trainer-engine-actor-{cell_index}"
        self.load_state_calls = 0

    async def load_state(self) -> list[int]:
        self.load_state_calls += 1
        return self.restored


def _guard(*, initialized: bool) -> InitOnce:
    guard = InitOnce("TrainerController")
    if initialized:
        with guard.guarding():
            pass
    return guard


def _make_controller(
    cells: list[_FakeCell], *, initialized: bool, expected_num_cells: int | None = None
) -> TrainerController:
    controller = object.__new__(TrainerController)
    controller._cells_by_id = {f"trainer-engine-actor-{cell.cell_index}": cell for cell in cells}
    controller._init_once = _guard(initialized=initialized)
    controller._role = "actor"
    controller._pool_id = compute_trainer_pool_id("actor")
    controller.args = SimpleNamespace(
        expected_num_cells=len(cells) if expected_num_cells is None else expected_num_cells
    )
    return controller


class TestTrainerControllerLoadState:
    async def test_every_cell_reloads_and_its_positions_reach_the_caller_in_cell_order(self):
        """Independent DP ranks are positional, so a reordered position list misreads which cell resumed where."""
        cells = [_FakeCell(cell_index=1, restored=[4, 4]), _FakeCell(cell_index=0, restored=[3, 3])]
        controller = _make_controller(cells, initialized=True)

        assert await controller.load_state() == [3, 3, 4, 4]
        assert [cell.load_state_calls for cell in cells] == [1, 1]

    async def test_a_controller_whose_cell_is_not_alive_refuses_to_reload_anything(self):
        """A take-over adopts the fleet a previous script left running; healing one is a different job."""
        cells = [_FakeCell(cell_index=0, restored=[3]), _FakeCell(cell_index=1, restored=[3], is_alive=False)]
        controller = _make_controller(cells, initialized=True)

        with pytest.raises(AssertionError, match="not alive"):
            await controller.load_state()

        assert [cell.load_state_calls for cell in cells] == [0, 0]

    async def test_a_controller_that_never_built_its_model_refuses_to_reload_it(self):
        """Loading a checkpoint into a model that was never built would read as a successful take-over."""
        controller = _make_controller([_FakeCell(cell_index=0, restored=[3])], initialized=False)

        with pytest.raises(AssertionError):
            await controller.load_state()


class TestTrainerCellLoadState:
    async def test_the_reload_reaches_every_worker_of_the_cell(self):
        """A worker left on its old weights would train a model the other ranks already rolled back."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert await cell.load_state() == [7, 7]

        for handle in get_raw_actor_handles(cell):
            assert [method for method, _args, _kwargs in ray.get(handle.get_calls.remote())] == ["load_state"]


class TestAFailedReloadLeavesTheCellStanding:
    async def test_a_worker_that_refuses_the_reload_does_not_kill_the_cell(self):
        """A take-over that cannot reload has to report that, not turn a live trainer into one to heal."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_fail_methods.remote(["load_state"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.load_state()

        assert cell.is_alive


class TestAReloadWaitsForTheWholeFleet:
    async def test_a_controller_still_short_of_a_cell_waits_rather_than_reloading_what_it_has(self, monkeypatch):
        """Reloading half a fleet would leave the missing cell holding the state the run has moved off."""
        monkeypatch.setattr(group_module, "_CELLS_READY_TIMEOUT_SECONDS", 0.05)
        cells = [_FakeCell(cell_index=0, restored=[3, 3])]
        controller = _make_controller(cells, initialized=True, expected_num_cells=2)

        with pytest.raises(TimeoutError, match="of 2 trainer cells observed"):
            await controller.load_state()

        assert [cell.load_state_calls for cell in cells] == [0]
