import asyncio
import logging

import httpx
import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import (
    RecordingHealthChecker,
    get_raw_actor_handles,
    make_alive_cell,
    make_cell,
    make_indep_dp_info,
)

from miles.ray.train.cell import _is_inference_engine_unreachable
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

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

    async def test_teardown_leaves_every_worker_handle_confirmed_dead(self):
        """Teardown really kills its workers, so every handle is confirmed dead and rejects new calls."""
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
        """Init configures the rendezvous address on every worker before dispatching init itself."""
        cell = make_cell(actor_count=2)
        info = make_indep_dp_info()

        results = await cell.init(indep_dp_info=info, indep_dp_store_addr="10.0.0.9:1234")

        assert len(results) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            assert [call[0] for call in calls] == ["configure_master_addr_and_port", "init"]
            assert calls[0][2] == {"master_addr": "10.0.0.1", "master_port": 20000}
            kwargs = calls[1][2]
            assert kwargs["indep_dp_info"] == info
            assert kwargs["indep_dp_store_addr"] == "10.0.0.9:1234"
            assert kwargs["recv_ckpt_src_rank"] is None

    async def test_health_checking_starts_on_an_alive_cell_and_is_running_when_init_returns(self):
        """A checker that is only scheduled never probes until the next await, which may be a whole train step away."""
        checker = RecordingHealthChecker()
        cell = make_cell(actor_count=1, health_checker=checker)
        checker.observe_alive = lambda: cell.is_alive

        await cell.init(indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)

        assert checker.start_count == 1
        assert checker.alive_when_started is True
        assert checker.task_started


class _EntryBarrier:
    def __init__(self, party_size: int) -> None:
        self._remaining = party_size
        self._all_entered = asyncio.Event()

    async def enter(self) -> None:
        self._remaining -= 1
        if self._remaining == 0:
            self._all_entered.set()
        await asyncio.wait_for(self._all_entered.wait(), timeout=10.0)


class _BarrierWorkerHandle:
    def __init__(self, barrier: _EntryBarrier) -> None:
        self._barrier = barrier

    async def train(self, **_kwargs) -> str:
        await self._barrier.enter()
        return "done"


class TestExecuteConcurrency:
    async def test_every_worker_rpc_is_entered_before_any_of_them_returns(self, monkeypatch: pytest.MonkeyPatch):
        """Worker methods enter a collective, so dispatching them one at a time deadlocks the cell."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        barrier = _EntryBarrier(3)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_BarrierWorkerHandle(barrier) for _ in range(3)])

        results = await cell.execute("train", rollout_id=0)

        assert results == ["done", "done", "done"]


class TestPartialWorkerFailure:
    async def test_one_failing_rank_errors_the_cell_and_kills_every_rank(self):
        """A surviving rank of a broken cell would sit in the collective and stall the cells that are still healthy."""
        cell = make_cell(actor_count=3)
        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())
        handles = get_raw_actor_handles(cell)
        ray.get(handles[1].set_fail_methods.remote(["train"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.execute("train", rollout_id=0)

        assert cell.is_errored
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_a_failure_that_must_not_kill_leaves_the_cell_alive_and_reachable(self):
        """The heartbeat probe rides on execute; recycling a cell because one probe raised would kill live training."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)
        ray.get(handles[0].set_fail_methods.remote(["train"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.execute("train", kill_on_failure=False, rollout_id=0)

        assert cell.is_alive
        assert not cell.is_errored
        for handle in handles:
            assert ray.get(handle.get_calls.remote())


class TestAsyncInitFailure:
    async def test_init_failure_leaves_cell_not_alive(self):
        """A failed remote init marks the cell errored and tears it down; it is never reported alive."""
        cell = make_cell(actor_count=1)
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_fail_methods.remote(["init"]))

        with pytest.raises(RuntimeError, match="Injected failure"):
            await cell.init(indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)

        assert not cell.is_alive
        for handle in get_raw_actor_handles(cell):
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())


class TestPrepareIndepDPModeAlive:
    async def test_reconfigure_and_update_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(
            indep_dp_info=new_info, indep_dp_store_addr="10.0.0.9:1234", send_ckpt_dst_ranks=[]
        )

        assert cell.indep_dp_info == new_info
        assert cell.is_alive

        for handle in get_raw_actor_handles(cell):
            calls = ray.get(handle.get_calls.remote())
            reconfig_calls = [c for c in calls if c[0] == "reconfigure_indep_dp"]
            assert len(reconfig_calls) == 1
            assert reconfig_calls[0][2]["indep_dp_info"] == new_info
            assert reconfig_calls[0][2]["indep_dp_store_addr"] == "10.0.0.9:1234"

    async def test_sends_ckpt_to_correct_dst_ranks(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=2)
        await cell.prepare_indep_dp_mode_alive(
            indep_dp_info=new_info, indep_dp_store_addr=None, send_ckpt_dst_ranks=[1, 2]
        )

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

        await cell.prepare_indep_dp_mode_healing(indep_dp_info=info, indep_dp_store_addr=None, recv_ckpt_src_rank=None)

        assert cell.is_alive
        assert cell.indep_dp_info == info

        handle = get_raw_actor_handles(cell)[0]
        calls = ray.get(handle.get_calls.remote())
        assert any(c[0] == "init" for c in calls)


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


class TestEngineConnectivityIsNotTheCellsFault:
    async def test_a_connect_error_to_an_engine_leaves_the_trainer_cell_alive(self):
        """The trainer is healthy; the engine it was told to broadcast to is the one that died."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [_HandleRaising(httpx.ConnectError("connection refused"))]

        with pytest.raises(httpx.ConnectError):
            await cell.execute("update_weights", info=None)

        assert cell.is_alive
        assert not cell.is_errored

    async def test_a_failure_of_the_cells_own_still_errors_and_kills_it(self):
        """Narrowing the blame must not stop a genuinely broken cell from being recycled."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [_HandleRaising(RuntimeError("optimizer step diverged"))]

        with pytest.raises(RuntimeError, match="optimizer step diverged"):
            await cell.execute("train", rollout_id=0)

        assert cell.is_errored


class TestIsInferenceEngineUnreachable:
    def test_a_direct_connect_error_is_recognised(self):
        """The trainer's own httpx client raises this when an engine's port stops accepting."""
        assert _is_inference_engine_unreachable(httpx.ConnectError("refused"))

    def test_a_connect_error_wrapped_by_the_rpc_boundary_is_recognised(self):
        """Ray re-raises the remote exception with the original one chained underneath."""
        cause = httpx.ConnectError("refused")
        error = RuntimeError("ray task error")
        error.__cause__ = cause

        assert _is_inference_engine_unreachable(error)

    def test_a_rebuilt_connect_error_class_is_recognised_by_name(self):
        """RayTaskError.as_instanceof_cause rebuilds the class, so identity with httpx's own can be lost."""
        rebuilt = type("ConnectError", (RuntimeError,), {})

        assert _is_inference_engine_unreachable(rebuilt("refused"))

    def test_an_unrelated_failure_is_not_blamed_on_an_engine(self):
        """Everything else is still the cell's own problem and must recycle it."""
        assert not _is_inference_engine_unreachable(RuntimeError("optimizer step diverged"))

    def test_a_cyclic_exception_chain_terminates(self):
        """__context__ can point back at an exception already seen, and a walk of it must not spin."""
        first, second = RuntimeError("a"), RuntimeError("b")
        first.__context__ = second
        second.__context__ = first

        assert not _is_inference_engine_unreachable(first)


class _HandleRaising(BaseWorkerHandle):
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __getattr__(self, name: str):
        async def call(**kwargs):
            raise self._error

        return call

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        return None

    async def probe_is_dead(self) -> bool:
        return False


class _SlowHandle(BaseWorkerHandle):
    def __init__(self, *, delay: float, error: BaseException | None = None) -> None:
        self._delay = delay
        self._error = error
        self.calls = 0

    def __getattr__(self, name: str):
        async def call(**kwargs):
            self.calls += 1
            await asyncio.sleep(self._delay)
            if self._error is not None:
                raise self._error
            return "ok"

        return call

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        return None

    async def probe_is_dead(self) -> bool:
        return False


class TestCellLivenessNeedsOnlyOneAnswer:
    async def test_a_worker_blocked_in_a_collective_does_not_time_out_the_cell_probe(self):
        """A cell whose other worker answers instantly is reachable, however long the blocked one takes."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        blocked = _SlowHandle(delay=3600)
        cell._state.worker_handles[:] = [blocked, _SlowHandle(delay=0)]

        await asyncio.wait_for(cell.probe_liveness(), timeout=1)

        assert cell.is_alive

    async def test_the_probe_still_fails_when_no_worker_answers(self):
        """An any-worker rule must not turn a whole dead cell into a healthy one."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [
            _SlowHandle(delay=0, error=WorkerUnreachableError("gone")),
            _SlowHandle(delay=0, error=WorkerUnreachableError("gone")),
        ]

        with pytest.raises(WorkerUnreachableError):
            await asyncio.wait_for(cell.probe_liveness(), timeout=1)

    async def test_a_dead_worker_alone_does_not_make_the_probe_fail(self):
        """Reporting that death is _scan_liveness_once's job (see test_ray_worker_manager_liveness.py); if this
        ever has to fail again, that scan is what stopped covering it."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [
            _SlowHandle(delay=0, error=WorkerUnreachableError("gone")),
            _SlowHandle(delay=0),
        ]

        await asyncio.wait_for(cell.probe_liveness(), timeout=1)

        assert cell.is_alive

    async def test_the_probe_does_not_leave_the_slow_worker_running(self):
        """A probe every ten seconds that leaks a task per blocked worker would pile them up all run."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [_SlowHandle(delay=3600), _SlowHandle(delay=0)]

        await asyncio.wait_for(cell.probe_liveness(), timeout=1)
        await asyncio.sleep(0)

        assert not [task for task in asyncio.all_tasks() if "get_heartbeat_status" in str(task.get_coro())]


class TestAFailedFanOutReportsEveryWorkersTime:
    async def test_the_failure_line_names_the_worker_that_never_answered(self, caplog):
        """A cell-level timing alone cannot say which worker was slow, which is what this bug cost us rounds on."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [
            _SlowHandle(delay=3600),
            _SlowHandle(delay=0, error=WorkerUnreachableError("gone")),
        ]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorkerUnreachableError):
                await cell.execute("train", kill_on_failure=False, rollout_id=0)

        assert "elapsed_s_of_worker=" in caplog.text
        assert "null" in caplog.text

    async def test_a_healthy_fan_out_logs_no_per_worker_line(self, caplog):
        """Every probe of every cell runs every ten seconds, so per-call records would bury the run's log."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._state.worker_handles[:] = [_SlowHandle(delay=0), _SlowHandle(delay=0)]

        with caplog.at_level(logging.INFO):
            await cell.execute("train", kill_on_failure=False, rollout_id=0)

        assert "elapsed_s_of_worker=" not in caplog.text
