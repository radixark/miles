import pytest
import ray

from tests.fast.ray.train.conftest import make_alive_cell, make_cell, make_indep_dp_info

pytestmark = pytest.mark.asyncio


class TestInitialState:
    def test_starts_as_uninitialized_after_init(self):
        """After __init__, cell is allocated (uninitialized) — actors created but not init'd."""
        cell = make_cell()

        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_pending

    def test_actor_handles_are_real_ray_actors(self):
        cell = make_cell(actor_count=3)

        handles = cell._get_actor_handles()
        assert len(handles) == 3
        assert all(isinstance(h, ray.actor.ActorHandle) for h in handles)


class TestMarkAsAlive:
    def test_transitions_uninitialized_to_alive(self):
        cell = make_cell()
        info = make_indep_dp_info(alive_cell_indices=[0, 1, 2])

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert cell.indep_dp_info == info

    def test_preserves_actor_handles(self):
        cell = make_cell(actor_count=3)
        handles_before = cell._get_actor_handles()

        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())

        assert cell._get_actor_handles() == handles_before

    def test_rejects_from_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell._mark_as_alive(indep_dp_info=make_indep_dp_info())


class TestInvalidTransitions:
    def test_allocate_for_pending_rejects_from_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell.allocate_for_pending()


class TestAsyncInit:
    async def test_dispatches_init_and_marks_alive(self):
        cell = make_cell(actor_count=2)
        info = make_indep_dp_info()

        results = await cell.init(indep_dp_info=info)

        assert len(results) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info

        for handle in cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert len(calls) == 1
            assert calls[0][0] == "init"
            kwargs = calls[0][2]
            assert kwargs["indep_dp_info"] == info


class TestStatePredicates:
    def test_uninitialized(self):
        cell = make_cell()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive

    def test_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert not cell.is_pending
        assert cell.is_allocated
        assert cell.is_alive
