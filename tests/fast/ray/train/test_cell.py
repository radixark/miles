import asyncio
import logging

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import (
    RecordingCellOperations,
    RecordingHealthChecker,
    get_raw_actor_handles,
    make_alive_cell,
    make_cell,
    make_indep_dp_info,
)

from miles.ray.train import cell as cell_module
from miles.utils.workers.cell_operations.base import CellTerminationNotConfirmedError, CellTerminationOutcome
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

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

    def test_a_cell_refuses_worker_infos_without_callable_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every described trainer rank must have a handle or the collective would silently omit it."""

        def _no_handles(_provider: RayWorkerProvider, _infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
            return {}

        monkeypatch.setattr(RayWorkerProvider, "get_handles_of_worker_infos", _no_handles)

        with pytest.raises(AssertionError, match="holds workers that cannot be called"):
            make_cell(actor_count=2)


class TestKillWorkers:
    async def test_a_restarted_worker_is_already_gone_for_killing(self):
        """A stale handle that reaches a new server must not make cell teardown fail."""

        class RestartedHandle:
            async def kill_self(self) -> None:
                raise ServerRestartedError

        await cell_module._kill_worker(RestartedHandle())

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


class TestMarkErroredAndKill:
    async def test_the_cell_is_errored_and_its_workers_are_gone(self):
        """A source the controller gave up on must stop being alive and stop holding its GPUs."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = get_raw_actor_handles(cell)

        await cell.mark_errored_and_kill("all of its weight update targets failed")

        assert cell.is_errored
        assert not cell.is_alive
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())


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


class _NeverReturningWorkerHandle:
    def __init__(self, *, dies_when_killed: bool = True) -> None:
        self.killed = False
        self.dead = False
        self.probes = 0
        self._dies_when_killed = dies_when_killed

    async def update_weights(self, **_kwargs) -> None:
        await asyncio.Event().wait()

    async def train(self, **_kwargs) -> str:
        await asyncio.Event().wait()

    async def kill_self(self) -> None:
        self.killed = True
        if not self._dies_when_killed:
            await asyncio.Event().wait()

    async def probe_is_dead(self) -> bool:
        self.probes += 1
        return self.dead or (self._dies_when_killed and self.killed)


class TestExecuteDeadline:
    """A worker that never answers has to be given up on, because nothing else can unblock the caller."""

    async def test_training_deadline_retires_a_blocked_cell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured training deadline retires a worker without waiting for heartbeat failure."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell.args.train_step_timeout = 0.05
        handle = _NeverReturningWorkerHandle()
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [handle])

        with pytest.raises(asyncio.TimeoutError):
            await cell.train(rollout_id=0, rollout_data_ref=None, witness_info=None, attempt=0)

        assert cell.is_errored
        assert handle.killed
        assert handle.probes > 0

    async def test_training_without_configured_deadline_preserves_cell_on_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """External cancellation does not turn an unlimited training attempt into a cell failure."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handle = _NeverReturningWorkerHandle()
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [handle])

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                cell.train(rollout_id=0, rollout_data_ref=None, witness_info=None, attempt=0), timeout=0.05
            )

        assert cell.is_alive
        assert not handle.killed

    async def test_a_wedged_worker_errors_the_cell_and_is_killed(self, monkeypatch: pytest.MonkeyPatch):
        """A trainer stuck in a native collective answers no heartbeat, so only the caller's deadline frees the run."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = [_NeverReturningWorkerHandle() for _ in range(2)]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(asyncio.TimeoutError):
            await cell.execute("update_weights", timeout=0.05, info=None, weight_version=1)

        assert cell.is_errored
        assert all(handle.killed for handle in handles)
        assert all(handle.probes > 0 for handle in handles)

    async def test_a_call_without_a_deadline_is_not_given_one(self, monkeypatch: pytest.MonkeyPatch):
        """Training steps and checkpoints legitimately run for hours; a borrowed deadline would kill them."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = [_NeverReturningWorkerHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cell.execute("train", rollout_id=0), timeout=0.05)

        assert not cell.is_errored
        assert not handles[0].killed

    async def test_a_worker_answering_inside_the_deadline_is_untouched(self, monkeypatch: pytest.MonkeyPatch):
        """The deadline must not disturb the normal path, which is every weight update that works."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        class _PromptWorkerHandle:
            async def update_weights(self, **_kwargs) -> int:
                return 7

        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_PromptWorkerHandle()])

        assert await cell.execute("update_weights", timeout=30.0, info=None, weight_version=1) == [7]
        assert cell.is_alive


class TestControlPlaneTermination:
    """kill_self runs inside the wedged process, so a cell that ignores it must be terminated from outside."""

    @pytest.fixture(autouse=True)
    def _impatient_confirmation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cell_module, "KILL_RPC_TIMEOUT_S", 0.05)
        monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.05)
        monkeypatch.setattr(cell_module, "DEAD_PROBE_INTERVAL_S", 0.01)
        monkeypatch.setattr(cell_module, "DEAD_PROBE_TIMEOUT_S", 0.05)
        monkeypatch.setattr(cell_module, "STALE_CONFIRM_TIMEOUT_S", 0.05)
        monkeypatch.setattr(cell_module, "TERMINATE_CALL_TIMEOUT_S", 0.05)

    async def test_a_worker_that_never_dies_is_terminated_through_the_control_plane(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole point: a process wedged in native code answers neither the rpc nor the probe."""
        operations = RecordingCellOperations()
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        handles = [_NeverReturningWorkerHandle(dies_when_killed=False) for _ in range(2)]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(asyncio.TimeoutError):
            await cell.execute("update_weights", timeout=0.05, info=None, weight_version=1)

        assert cell.is_errored
        assert all(handle.killed for handle in handles)
        assert operations.terminated == [(cell.cell_id, "pseudo-hash-1")]

    async def test_a_worker_that_dies_on_request_is_not_terminated_through_the_control_plane(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The rpc is the fast path; going to the control plane for every failure would be a pointless api call."""
        operations = RecordingCellOperations()
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        handles = [_NeverReturningWorkerHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(asyncio.TimeoutError):
            await cell.execute("update_weights", timeout=0.05, info=None, weight_version=1)

        assert operations.terminated == []

    async def test_a_successful_weight_update_terminates_nothing(self, monkeypatch: pytest.MonkeyPatch):
        """Killing a healthy trainer because the control plane call is now wired up would end the run."""
        operations = RecordingCellOperations()
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)

        class _PromptWorkerHandle:
            async def update_weights(self, **_kwargs) -> int:
                return 7

        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_PromptWorkerHandle()])

        assert await cell.execute("update_weights", timeout=30.0, info=None, weight_version=1) == [7]
        assert operations.terminated == []

    async def test_a_stale_answer_alone_does_not_confirm_the_workers_died(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A kubernetes hash moves when the pod is merely marked for deletion, while its process runs on."""
        operations = RecordingCellOperations()
        operations.outcome = CellTerminationOutcome.STALE
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_NeverReturningWorkerHandle(dies_when_killed=False)])

        with caplog.at_level(logging.INFO, logger=cell_module.__name__):
            await cell._kill_workers_and_confirm_dead()

        assert operations.terminated == [(cell.cell_id, "pseudo-hash-1")]
        assert "confirmed=false" in caplog.text
        assert "confirmed=true" not in caplog.text

    async def test_a_stale_answer_is_confirmed_once_the_pinned_handles_stop_answering(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A cell somebody else already removed is confirmed by probing the handles pinned to that incarnation."""
        handle = _NeverReturningWorkerHandle(dies_when_killed=False)

        class _CellOperationsThatFoundItAlreadyRemoved(RecordingCellOperations):
            async def terminate_incarnation(self, **kwargs) -> CellTerminationOutcome:
                handle.dead = True
                return await super().terminate_incarnation(**kwargs)

        operations = _CellOperationsThatFoundItAlreadyRemoved()
        operations.outcome = CellTerminationOutcome.STALE
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [handle])

        with caplog.at_level(logging.INFO, logger=cell_module.__name__):
            await cell._kill_workers_and_confirm_dead()

        assert operations.terminated == [(cell.cell_id, "pseudo-hash-1")]
        assert "confirmed=true" in caplog.text

    async def test_a_control_plane_that_never_answers_leaves_the_cell_unconfirmed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The control plane can be wedged too, and waiting on it forever is the deadlock this op removes."""
        operations = RecordingCellOperations()
        operations.hangs = True
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_NeverReturningWorkerHandle(dies_when_killed=False)])

        with caplog.at_level(logging.INFO, logger=cell_module.__name__):
            await asyncio.wait_for(cell._kill_workers_and_confirm_dead(), timeout=5.0)

        assert "confirmed=false" in caplog.text

    async def test_a_failed_termination_is_never_reported_as_confirmed_dead(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A run that believes a wedged rank died would restart its dp group while the old rank still holds the gpu."""
        operations = RecordingCellOperations()
        operations.error = CellTerminationNotConfirmedError("the pod is still there")
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=operations)
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_NeverReturningWorkerHandle(dies_when_killed=False)])

        with caplog.at_level(logging.INFO, logger=cell_module.__name__):
            await cell._kill_workers_and_confirm_dead()

        assert "confirmed=false" in caplog.text
        assert "confirmed=true" not in caplog.text

    async def test_a_confirmed_termination_is_reported_as_confirmed_dead(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The healing path reads this log to tell a released cell from one still holding its gpus."""
        cell = make_alive_cell(0, alive_cell_indices=[0], cell_operations=RecordingCellOperations())
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: [_NeverReturningWorkerHandle(dies_when_killed=False)])

        with caplog.at_level(logging.INFO, logger=cell_module.__name__):
            await cell._kill_workers_and_confirm_dead()

        assert "confirmed=true" in caplog.text


class TestWaitAllWorkersDead:
    @pytest.fixture(autouse=True)
    def _fast_probes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cell_module, "DEAD_PROBE_INTERVAL_S", 0.01)

    async def test_a_worker_that_never_answers_the_probe_is_not_confirmed_dead(self):
        """wait_dead returns after logging a timeout, which would pass a live worker off as a dead one."""
        handle = _NeverReturningWorkerHandle(dies_when_killed=False)

        assert not await cell_module._wait_all_workers_dead([handle], timeout=0.05)

    async def test_every_worker_must_be_dead_before_the_cell_is(self):
        """One surviving rank keeps the collective alive, so a partly dead cell is not confirmed dead."""
        dead, alive = _NeverReturningWorkerHandle(), _NeverReturningWorkerHandle(dies_when_killed=False)
        await dead.kill_self()

        assert not await cell_module._wait_all_workers_dead([dead, alive], timeout=0.05)

    async def test_workers_that_died_are_confirmed(self):
        """The common case must stay cheap: dead workers are confirmed on the first probe."""
        handles = [_NeverReturningWorkerHandle() for _ in range(2)]
        for handle in handles:
            await handle.kill_self()

        assert await cell_module._wait_all_workers_dead(handles, timeout=30.0)
        assert all(handle.probes == 1 for handle in handles)

    async def test_a_probe_that_never_answers_counts_as_still_running(self, monkeypatch: pytest.MonkeyPatch):
        """A probe against a wedged node can hang as hard as the call it is checking on."""
        monkeypatch.setattr(cell_module, "DEAD_PROBE_TIMEOUT_S", 0.01)

        class _SilentProbeHandle:
            async def probe_is_dead(self) -> bool:
                await asyncio.Event().wait()

        assert not await asyncio.wait_for(
            cell_module._wait_all_workers_dead([_SilentProbeHandle()], timeout=0.05), timeout=5.0
        )

    async def test_a_probe_that_raises_counts_as_still_running(self):
        """An unreachable probe is not evidence of death, and treating it as one would skip the real termination."""

        class _BrokenProbeHandle:
            async def probe_is_dead(self) -> bool:
                raise RuntimeError("the node is unreachable")

        assert not await cell_module._wait_all_workers_dead([_BrokenProbeHandle()], timeout=0.05)
