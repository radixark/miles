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
        assert cell.is_stopped

    async def test_restarting_asks_the_manager_to_start_the_cell(self):
        """After a fault the actors are gone, so somebody must recreate them."""
        cell = make_cell(2)
        await cell.stop()
        cell.mark_as_pending()

        await cell.allocate_for_pending()

        assert train_conftest.fake_worker_manager.started_cell_ids == [["trainer-actor-2"]]
        assert cell.is_allocated

    async def test_restarting_picks_up_the_fresh_actor_handles(self):
        """Reusing the dead handles would make every later call fail."""
        cell = make_cell(0)
        old_handles = cell._get_actor_handles()
        await cell.stop()
        cell.mark_as_pending()

        await cell.allocate_for_pending()

        assert cell._get_actor_handles() != old_handles

    async def test_stopping_twice_does_not_ask_the_manager_again(self):
        """Healing may stop an already dead cell, which the manager would reject."""
        cell = make_cell(0)
        await cell.stop()

        await cell.stop()

        assert len(train_conftest.fake_worker_manager.stopped_cell_ids) == 1
