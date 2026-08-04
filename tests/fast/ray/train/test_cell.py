import asyncio

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell, make_cell, make_indep_dp_info

from miles.utils.workers.worker_handle import BaseWorkerHandle

pytestmark = pytest.mark.asyncio


class TestInitialState:
    def test_starts_as_uninitialized_after_init(self):
        """After __init__, cell is allocated (uninitialized) — actors created but not init'd."""
        cell = make_cell()

        assert cell.is_allocated
        assert not cell.is_alive
        assert cell.is_uninitialized

    def test_worker_handles_wrap_real_ray_actors(self):
        cell = make_cell(actor_count=3)

        handles = cell._get_worker_handles()
        assert len(handles) == 3
        assert all(isinstance(h, BaseWorkerHandle) for h in handles)
        assert all(isinstance(h, ray.actor.ActorHandle) for h in get_raw_actor_handles(cell))


class TestKillWorkers:
    async def test_killing_reaches_every_worker(self):
        """The dead workers must not linger in a cross-cell collective."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_killing_does_not_involve_the_worker_manager(self):
        """The manager keeps reporting the cell alive so its errored status stays visible."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        await cell._kill_workers_and_confirm_dead()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == []

    async def test_stop_kills_the_underlying_workers(self):
        """Stopping a cell really kills its workers, so every handle is confirmed dead and rejects new calls."""
        cell = make_cell(actor_count=2)
        wrapped_handles = cell._get_worker_handles()

        await cell._kill_workers_and_confirm_dead()

        for wrapped in wrapped_handles:
            await asyncio.wait_for(wrapped.wait_dead(timeout=30.0), timeout=35.0)
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(wrapped._actor_handle.get_calls.remote())


class TestMarkAsAlive:
    def test_transitions_uninitialized_to_alive(self):
        cell = make_cell()
        info = make_indep_dp_info(alive_cell_indices=[0, 1, 2])

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert cell.indep_dp_info == info

    def test_preserves_actor_handles(self):
        cell = make_cell(actor_count=3)
        handles_before = get_raw_actor_handles(cell)

        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())

        assert get_raw_actor_handles(cell) == handles_before

    def test_rejects_from_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell._mark_as_alive(indep_dp_info=make_indep_dp_info())


class TestUpdateIndepDPInfo:
    def test_updates_stored_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info

    def test_preserves_actor_handles(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)

        cell._update_indep_dp_info(make_indep_dp_info(quorum_id=5))

        assert get_raw_actor_handles(cell) == handles

    def test_rejects_from_uninitialized(self):
        cell = make_cell()

        with pytest.raises(AssertionError):
            cell._update_indep_dp_info(make_indep_dp_info())


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

    def test_transitions_uninitialized_to_errored_without_info(self):
        """A cell whose init never completed can still be marked errored; its indep_dp_info is None."""
        cell = make_cell()

        cell._mark_as_errored()

        assert cell.is_errored
        assert cell.indep_dp_info is None


class TestErroredCellTeardown:
    async def test_kill_from_errored_reaches_the_workers(self):
        """An errored cell is torn down by killing its own workers."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()
        assert cell.is_errored
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_the_replacement_cell_recovers_the_lifecycle(self):
        """Errored → kill → heal restarts the cell → reconcile builds a fresh one → alive."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()
        await cell._kill_workers_and_confirm_dead()

        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        replacement = make_cell(cell.cell_index)
        replacement._mark_as_alive(indep_dp_info=make_indep_dp_info(quorum_id=99))

        assert replacement.is_alive
        assert replacement.indep_dp_info.quorum_id == 99


class TestAsyncInit:
    async def test_dispatches_init_and_marks_alive(self):
        cell = make_cell(actor_count=2)
        info = make_indep_dp_info()

        results = await cell.init(indep_dp_info=info)

        assert len(results) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            assert [name for name, _args, _kwargs in calls] == ["configure_master_addr_and_port", "init"]
            kwargs = calls[1][2]
            assert kwargs["indep_dp_info"] == info
            assert kwargs["recv_ckpt_src_rank"] is None


class TestAsyncInitFailure:
    async def test_init_failure_leaves_cell_not_alive(self):
        """A failed remote init marks the cell errored and tears it down; it is never reported alive."""
        cell = make_cell(actor_count=1)
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_fail_methods.remote(["init"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.init(indep_dp_info=make_indep_dp_info())

        assert not cell.is_alive
        for handle in get_raw_actor_handles(cell):
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())


class TestPrepareIndepDPModeAlive:
    async def test_reconfigure_and_update_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[])

        assert cell.indep_dp_info == new_info
        assert cell.is_alive

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            reconfig_calls = [c for c in calls if c[0] == "reconfigure_indep_dp"]
            assert len(reconfig_calls) == 1
            assert reconfig_calls[0][2]["indep_dp_info"] == new_info

    async def test_sends_ckpt_to_correct_dst_ranks(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[1, 2])

        handle = get_raw_actor_handles(cell)[0]
        calls = ray.get(handle.get_calls.remote())
        send_calls = [c for c in calls if c[0] == "send_ckpt"]
        assert len(send_calls) == 2
        assert send_calls[0][2]["dst_rank"] == 1
        assert send_calls[1][2]["dst_rank"] == 2


class TestPrepareIndepDPModeHealing:
    async def test_healing_inits_and_marks_alive(self):
        cell = make_cell(actor_count=1)
        info = make_indep_dp_info()

        await cell.prepare_indep_dp_mode_healing(indep_dp_info=info, recv_ckpt_src_rank=None)

        assert cell.is_alive
        assert cell.indep_dp_info == info

        handle = get_raw_actor_handles(cell)[0]
        calls = ray.get(handle.get_calls.remote())
        assert any(c[0] == "init" for c in calls)


class TestSetRolloutExecutor:
    async def test_missing_rollout_executor_skips_worker_rpc(self):
        """A cell configured without a rollout executor returns an empty result and dispatches nothing."""
        cell = make_cell(actor_count=2)

        results = await cell.set_rollout_executor()

        assert results == []
        for handle in cell._get_actor_handles():
            assert ray.get(handle.get_calls.remote()) == []

    async def test_present_rollout_executor_reaches_every_actor(self):
        """A configured rollout executor handle is forwarded positionally to every actor of the cell."""
        cell = make_cell(actor_count=2, rollout_executor="executor-handle")

        results = await cell.set_rollout_executor()

        assert len(results) == 2
        for handle in cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert calls == [("set_rollout_executor", ("executor-handle",), {})]


class TestStatePredicates:
    def test_uninitialized(self):
        cell = make_cell()

        assert cell.is_allocated
        assert cell.is_uninitialized
        assert not cell.is_alive
        assert not cell.is_errored

    def test_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert cell.is_allocated
        assert not cell.is_uninitialized
        assert cell.is_alive
        assert not cell.is_errored

    def test_errored(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        assert cell.is_allocated
        assert not cell.is_uninitialized
        assert not cell.is_alive
        assert cell.is_errored


class TestFullLifecycle:
    async def test_full_kill_and_replacement_cycle(self):
        """Full lifecycle: attach → alive → kill → heal restarts → reconcile replaces the object → alive again."""
        # Step 1: Create (attaches to the manager's workers)
        cell = make_cell(actor_count=2)
        assert cell.is_uninitialized and not cell.is_alive

        # Step 2: Alive
        info_v1 = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=1)
        cell._mark_as_alive(indep_dp_info=info_v1)
        assert cell.is_alive

        # Step 3: Kill the workers directly
        await cell._kill_workers_and_confirm_dead()

        # Step 4: The ft controller heals it and reconcile builds a fresh object on the new workers
        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        cell = make_cell(actor_count=2)
        assert cell.is_uninitialized and not cell.is_alive

        # Step 5: Alive again with new config
        info_v2 = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._mark_as_alive(indep_dp_info=info_v2)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 2
