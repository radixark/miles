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
        assert not cell.is_stopped

    def test_actor_handles_are_real_ray_actors(self):
        cell = make_cell(actor_count=3)

        handles = cell._get_actor_handles()
        assert len(handles) == 3
        assert all(isinstance(h, ray.actor.ActorHandle) for h in handles)


class TestStopTransitions:
    def test_stop_from_uninitialized_kills_actors(self):
        cell = make_cell(actor_count=2)

        cell.stop()

        assert cell.is_stopped
        assert not cell.is_allocated

    def test_stop_from_alive_kills_actors(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        cell.stop()

        assert cell.is_stopped

    def test_stop_from_pending_transitions_to_stopped(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        cell.stop()

        assert cell.is_stopped

    def test_stop_already_stopped_is_idempotent(self):
        cell = make_cell()
        cell.stop()

        cell.stop()

        assert cell.is_stopped


class TestMarkAsPending:
    def test_from_stopped(self):
        cell = make_cell()
        cell.stop()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_pending(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_allocated(self):
        cell = make_cell()

        cell.mark_as_pending()

        assert cell.is_allocated


class TestAllocateForPending:
    def test_reallocate_after_stop_start(self):
        """After stop → pending → allocate, cell has fresh actors."""
        cell = make_cell(actor_count=2)
        old_handles = cell._get_actor_handles()

        cell.stop()
        cell.mark_as_pending()
        cell.allocate_for_pending()

        assert cell.is_allocated
        new_handles = cell._get_actor_handles()
        assert len(new_handles) == 2
        assert new_handles != old_handles


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


class TestMarkAsErrored:
    def test_transitions_alive_to_errored(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        info = cell.indep_dp_info

        cell._mark_as_errored()

        assert cell.is_errored
        assert not cell.is_alive
        assert cell.is_allocated
        assert cell.indep_dp_info == info

    def test_errored_is_idempotent(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        cell._mark_as_errored()

        assert cell.is_errored


class TestInvalidTransitions:
    def test_mark_as_errored_rejects_from_uninitialized(self):
        cell = make_cell()

        with pytest.raises(AssertionError):
            cell._mark_as_errored()

    def test_allocate_for_pending_rejects_from_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell.allocate_for_pending()

    def test_allocate_for_pending_rejects_from_stopped(self):
        cell = make_cell()
        cell.stop()

        with pytest.raises(AssertionError):
            cell.allocate_for_pending()


class TestErroredToStopped:
    def test_stop_from_errored_transitions_to_stopped(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()
        assert cell.is_errored

        cell.stop()

        assert cell.is_stopped
        assert not cell.is_errored

    def test_full_error_recovery_lifecycle(self):
        """Errored → stop → pending → allocate → alive (full recovery from error)."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        cell.stop()
        cell.mark_as_pending()
        cell.allocate_for_pending()
        cell._mark_as_alive(indep_dp_info=make_indep_dp_info(quorum_id=99))

        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 99


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
    def test_pending(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        assert cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_uninitialized(self):
        cell = make_cell()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert not cell.is_pending
        assert cell.is_allocated
        assert cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_errored(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert cell.is_errored
        assert not cell.is_stopped

    def test_stopped(self):
        cell = make_cell()
        cell.stop()

        assert not cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert cell.is_stopped


class TestFullLifecycle:
    def test_full_stop_start_cycle(self):
        """Full lifecycle: init → alive → stop → pending → allocate → alive."""
        # Step 1: Create (Pending → Uninitialized)
        cell = make_cell(actor_count=2)
        assert cell.is_allocated and not cell.is_alive

        # Step 2: Alive
        info_v1 = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=1)
        cell._mark_as_alive(indep_dp_info=info_v1)
        assert cell.is_alive

        # Step 3: Stop
        cell.stop()
        assert cell.is_stopped

        # Step 4: Pending
        cell.mark_as_pending()
        assert cell.is_pending

        # Step 5: Allocate (new actors)
        cell.allocate_for_pending()
        assert cell.is_allocated and not cell.is_alive

        # Step 6: Alive again with new config
        info_v2 = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._mark_as_alive(indep_dp_info=info_v2)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 2
        assert cell.indep_dp_info.alive_size == 2
