import asyncio
import logging
from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell, make_cell, make_indep_dp_info

from miles.ray.train import cell as cell_module

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


class TestUpdateIndepDPInfo:
    def test_updates_stored_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info

    def test_preserves_actor_handles(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = cell._get_actor_handles()

        cell._update_indep_dp_info(make_indep_dp_info(quorum_id=5))

        assert cell._get_actor_handles() == handles

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


class TestInvalidTransitions:
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
            assert kwargs["recv_ckpt_src_rank"] is None


class TestAsyncInitFailure:
    async def test_init_failure_leaves_cell_stopped_not_alive(self):
        """A failed remote init routes through errored to stopped; the cell is never reported alive."""
        cell = make_cell(actor_count=1)
        for handle in cell._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["init"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.init(indep_dp_info=make_indep_dp_info())

        assert not cell.is_alive
        assert cell.is_stopped


class TestPrepareIndepDPModeAlive:
    async def test_reconfigure_and_update_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[])

        assert cell.indep_dp_info == new_info
        assert cell.is_alive

        for handle in cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            reconfig_calls = [c for c in calls if c[0] == "reconfigure_indep_dp"]
            assert len(reconfig_calls) == 1
            assert reconfig_calls[0][2]["indep_dp_info"] == new_info

    async def test_sends_ckpt_to_correct_dst_ranks(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[1, 2])

        handle = cell._get_actor_handles()[0]
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

        handle = cell._get_actor_handles()[0]
        calls = ray.get(handle.get_calls.remote())
        assert any(c[0] == "init" for c in calls)


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


def _make_coro_factory(behavior):
    async def _coro():
        return behavior()

    return _coro


def _raise_factory(exc):
    async def _coro():
        raise exc

    return _coro


class _FakeReadyMethod:
    def __init__(self, coro_factories):
        self._coro_factories = list(coro_factories)
        self.call_count = 0

    def remote(self):
        self.call_count += 1
        index = min(self.call_count - 1, len(self._coro_factories) - 1)
        return self._coro_factories[index]()


def _make_fake_handle(coro_factories):
    ready = _FakeReadyMethod(coro_factories)
    handle = SimpleNamespace()
    handle.__ray_ready__ = ready
    return handle, ready


def _ray_actor_error():
    return ray.exceptions.RayActorError()


def _ray_task_error():
    return ray.exceptions.RayTaskError.__new__(ray.exceptions.RayTaskError)


class TestConfirmActorDead:
    async def test_returns_immediately_when_actor_error_on_first_probe(self):
        """A dead actor whose first probe raises RayActorError is confirmed dead after one probe."""
        handle, ready = _make_fake_handle([_raise_factory(_ray_actor_error())])

        await cell_module._confirm_actor_dead(handle)

        assert ready.call_count == 1

    async def test_returns_immediately_when_task_error_on_first_probe(self):
        """RayTaskError on the first probe is also treated as confirmed actor death."""
        handle, ready = _make_fake_handle([_raise_factory(_ray_task_error())])

        await cell_module._confirm_actor_dead(handle)

        assert ready.call_count == 1

    async def test_retries_after_timeout_then_confirms_death(self, monkeypatch):
        """A probe timeout is tolerated; the loop retries and confirms death on the next probe."""
        slept = []

        async def _noop_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(cell_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(cell_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 1.0])))

        handle, ready = _make_fake_handle(
            [
                _raise_factory(asyncio.TimeoutError()),
                _raise_factory(_ray_actor_error()),
            ]
        )

        await cell_module._confirm_actor_dead(handle)

        assert ready.call_count == 2
        assert slept == [1.0]

    async def test_deadline_reached_returns_and_logs_error(self, monkeypatch, caplog):
        """When the timeout deadline is exceeded after a hung probe, it returns and logs an ERROR."""

        async def _noop_sleep(seconds):
            return None

        monkeypatch.setattr(cell_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(cell_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 200.0])))

        handle, ready = _make_fake_handle([_raise_factory(asyncio.TimeoutError())])

        with caplog.at_level(logging.ERROR, logger="miles.ray.train.cell"):
            await cell_module._confirm_actor_dead(handle)

        assert ready.call_count == 1
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "Timed out after 120s confirming actor death" in error_records[0].getMessage()


def _make_monotonic(values):
    seq = list(values)
    state = {"i": 0}

    def _monotonic():
        i = state["i"]
        if i < len(seq):
            state["i"] = i + 1
            return seq[i]
        return seq[-1]

    return _monotonic
