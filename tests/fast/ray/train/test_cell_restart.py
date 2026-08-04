import pytest
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import make_cell

pytestmark = pytest.mark.asyncio


class TestCellRestartGoesThroughTheManager:
    async def test_stopping_asks_the_manager_instead_of_killing_actors(self):
        """The manager owns the actors, so a cell killing them behind its back desyncs it."""
        cell = make_cell(2)

        await cell.stop()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == [["trainer-actor-2"]]

    async def test_a_replacement_cell_picks_up_the_fresh_actor_handles(self):
        """Reusing the dead handles would make every later call fail."""
        cell = make_cell(0)
        old_handles = cell._get_actor_handles()
        await cell.stop()

        replacement = make_cell(0)

        assert replacement._get_actor_handles() != old_handles

    async def test_stopping_twice_asks_the_manager_twice(self):
        """The cell no longer tracks a stopped state, so idempotence is the manager's job."""
        cell = make_cell(0)
        await cell.stop()

        await cell.stop()

        assert len(train_conftest.fake_worker_manager.stopped_cell_ids) == 2
