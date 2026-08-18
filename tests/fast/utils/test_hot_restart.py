from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from argparse import Namespace
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.train.group import TrainerController
from miles.utils import hot_restart as hot_restart_module
from miles.utils.context_lock import ContextLock
from miles.utils.hot_restart import (
    init_or_reset_inference_controller,
    trainer_init_or_load_state,
    wait_trainers_idle,
    wait_until_worker_not_initialized,
)
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import NamedHostAndPorts

_TRAINER_ID = "policy_a-actor"
_OTHER_TRAINER_ID = "policy_b-actor"
_STALLED_SECONDS = 5.0
_SHORT_BUDGET_SECONDS = 0.05
_BROADCAST_ARGS = Namespace(update_weight_transfer_mode="broadcast")
_DISK_DELTA_ARGS = Namespace(update_weight_transfer_mode="disk-delta")


class _FakeTrainer:
    def __init__(
        self,
        *,
        initialized: bool,
        idle_seconds: float = 0.0,
        load_seconds: float = 0.0,
        trainer_id: str = _TRAINER_ID,
        fleet_calls: list[str] | None = None,
    ) -> None:
        self.initialized = initialized
        self.idle_seconds = idle_seconds
        self.load_seconds = load_seconds
        self.trainer_id = trainer_id
        self.fleet_calls = fleet_calls
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        self._record("is_initialized")
        return self.initialized

    async def init(self, model_args: Namespace) -> list[Any]:
        self._record("init")
        return [7]

    async def load_state(self) -> list[Any]:
        self._record("load_state")
        await asyncio.sleep(self.load_seconds)
        return [3]

    async def wait_idle(self, *, timeout: float) -> None:
        self._record("wait_idle")
        self.idle_timeouts.append(timeout)
        await asyncio.wait_for(asyncio.sleep(self.idle_seconds), timeout=timeout)

    def _record(self, call: str) -> None:
        self.calls.append(call)
        if self.fleet_calls is not None:
            self.fleet_calls.append(f"{self.trainer_id}.{call}")


class _FakeInferenceController:
    def __init__(
        self,
        *,
        initialized: bool,
        busy: bool = False,
        wedged: bool = False,
        abort_error: Exception | None = None,
        fleet_incomplete: bool = False,
    ) -> None:
        self.initialized = initialized
        self.busy = busy
        self.wedged = wedged
        self.abort_error = abort_error
        self.fleet_incomplete = fleet_incomplete
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []
        self.fleet_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        self.calls.append("is_initialized")
        return self.initialized

    async def init(self) -> None:
        self.calls.append("init")

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        if self.busy:
            raise TimeoutError("InferenceController was still busy")

    async def wait_expected_num_cells(self, timeout: float) -> None:
        self.calls.append("wait_expected_num_cells")
        self.fleet_timeouts.append(timeout)
        if self.fleet_incomplete:
            raise TimeoutError("the fleet is short of engines")

    async def abort_all(self) -> None:
        self.calls.append("abort_all")
        await self._maybe_hang()
        if self.abort_error is not None:
            raise self.abort_error

    async def _maybe_hang(self) -> None:
        if self.wedged:
            await asyncio.sleep(_STALLED_SECONDS)


class _FakeWorker:
    def __init__(self, answers: list[bool | Exception], *, ready_errors: list[Exception] | None = None) -> None:
        self._answers = list(answers)
        self._ready_errors = list(ready_errors or [])
        self.ready_timeouts: list[float] = []
        self.ready_allowances: list[bool] = []

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        self.ready_timeouts.append(timeout)
        self.ready_allowances.append(allow_server_uuid_change)
        if self._ready_errors:
            raise self._ready_errors.pop(0)

    async def is_initialized(self) -> bool:
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _StuckThenNotInitializedWorker:
    def __init__(self) -> None:
        self.rounds = 0

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        return None

    async def is_initialized(self) -> bool:
        self.rounds += 1
        if self.rounds == 1:
            await asyncio.sleep(30.0)
        return False


@pytest.fixture
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "_WORKER_POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture
def short_take_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "TAKE_OVER_GATE_TIMEOUT_SECONDS", _SHORT_BUDGET_SECONDS)


@pytest.fixture
def short_reload_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "_TRAINER_RELOAD_TIMEOUT_SECONDS", _SHORT_BUDGET_SECONDS)


class TestWaitUntilWorkerNotInitialized:
    async def test_a_fresh_worker_is_accepted_at_once(self):
        """The normal case must not pay a polling delay."""
        worker = _FakeWorker([False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_a_worker_of_the_previous_script_is_waited_out(self, fast_polling: None):
        """Initializing the old worker a second time would drive the process that is about to die."""
        worker = _FakeWorker([True, True, False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_the_poll_interval_paces_the_wait(self, monkeypatch: pytest.MonkeyPatch):
        """A busy loop against a worker that answers at once would hammer it for the whole budget."""
        monkeypatch.setattr(hot_restart_module, "_WORKER_POLL_INTERVAL_SECONDS", 0.2)
        worker = _FakeWorker([True, False])

        started = time.monotonic()
        await wait_until_worker_not_initialized(worker, timeout=5.0)

        assert time.monotonic() - started >= 0.2

    async def test_every_attempt_lets_the_worker_process_be_replaced(self, fast_polling: None):
        """The worker is expected to restart during this wait, so its boot uuid change is not a violation."""
        worker = _FakeWorker([True, False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

        assert worker.ready_allowances == [True, True]

    async def test_a_worker_replaced_mid_wait_is_waited_out(self, fast_polling: None):
        """The replacement answers a call the old process accepted, which is the restart this wait exists for."""
        worker = _FakeWorker([True, ServerRestartedError("replaced"), False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_a_worker_that_stays_initialized_gives_up_loudly(self, fast_polling: None):
        """A new script must not silently share a run with the worker of its predecessor."""
        worker = _FakeWorker([True] * 100)

        with pytest.raises(hot_restart_module._StillInitializedError, match="already initialized it"):
            await wait_until_worker_not_initialized(worker, timeout=_SHORT_BUDGET_SECONDS)

    async def test_a_worker_pod_being_recreated_is_waited_out(self, fast_polling: None):
        """Replacing the pod is the whole point, and it is unreachable for as long as that takes."""
        worker = _FakeWorker([True, WorkerUnreachableError("pod is gone"), False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_a_transport_error_is_waited_out_too(self, fast_polling: None):
        """A connection refused while kubernetes reschedules the pod is the expected state, not a failure."""
        worker = _FakeWorker([httpx.ConnectError("refused"), False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_a_worker_that_stays_unreachable_raises_what_it_last_saw(self, fast_polling: None):
        """A wait that ends without the worker ever answering has to name the failure it kept hitting."""
        worker = _FakeWorker([WorkerUnreachableError("pod is gone")] * 100)

        with pytest.raises(WorkerUnreachableError, match="pod is gone"):
            await wait_until_worker_not_initialized(worker, timeout=_SHORT_BUDGET_SECONDS)

    async def test_a_readiness_wait_that_fails_is_retried_rather_than_fatal(self, fast_polling: None):
        """wait_ready itself throws while the replacement pod is still being scheduled."""
        worker = _FakeWorker([False], ready_errors=[WorkerUnreachableError("no")])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_a_readiness_wait_that_reports_yet_another_restart_is_retried(self, fast_polling: None):
        """Two pod replacements in a row are still an ordinary hot restart, not a violation to raise on."""
        worker = _FakeWorker([False], ready_errors=[ServerRestartedError("again")])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

    async def test_the_readiness_probe_gets_its_own_budget(self, fast_polling: None):
        """Handing it the poll interval would give a pod being scheduled a few seconds to come back."""
        worker = _FakeWorker([True, False])

        await wait_until_worker_not_initialized(worker, timeout=5.0)

        assert worker.ready_timeouts == [hot_restart_module._WORKER_READY_TIMEOUT_SECONDS] * 2

    async def test_a_round_that_never_answers_is_cut_short(self, fast_polling: None, monkeypatch: pytest.MonkeyPatch):
        """A worker that accepts the call and never answers would eat the whole budget in one round."""
        monkeypatch.setattr(hot_restart_module, "_WORKER_POLL_ATTEMPT_TIMEOUT_SECONDS", 0.02)
        worker = _StuckThenNotInitializedWorker()

        await wait_until_worker_not_initialized(worker, timeout=5.0)

        assert worker.rounds == 2


class TestTheTrainersAreWaitedIdle:
    async def test_a_trainer_that_never_ran_is_not_waited_for(self):
        """A cold start must be untouched by the resume protocol, and it is running nothing to wait out."""
        trainer = _FakeTrainer(initialized=False)

        assert await wait_trainers_idle({_TRAINER_ID: trainer}) is False

        assert trainer.calls == ["is_initialized"]

    async def test_a_surviving_trainer_is_waited_out_before_anything_drives_it(self):
        """A previous script may still be running a train step on it, and reloading over that would corrupt it."""
        trainer = _FakeTrainer(initialized=True)

        assert await wait_trainers_idle({_TRAINER_ID: trainer}) is True

        assert trainer.calls == ["is_initialized", "wait_idle"]

    async def test_the_whole_fleet_is_asked_before_any_of_it_is_drained(self):
        """A fleet that disagrees must stop the run before it spends the budget draining half of it."""
        fleet_calls: list[str] = []
        first = _FakeTrainer(initialized=True, trainer_id=_TRAINER_ID, fleet_calls=fleet_calls)
        second = _FakeTrainer(initialized=True, trainer_id=_OTHER_TRAINER_ID, fleet_calls=fleet_calls)

        await wait_trainers_idle({_TRAINER_ID: first, _OTHER_TRAINER_ID: second})

        assert fleet_calls == [
            f"{_TRAINER_ID}.is_initialized",
            f"{_OTHER_TRAINER_ID}.is_initialized",
            f"{_TRAINER_ID}.wait_idle",
            f"{_OTHER_TRAINER_ID}.wait_idle",
        ]

    async def test_a_disagreeing_fleet_is_refused_before_anything_is_drained(self):
        """Draining a fleet that cannot be taken over spends 600s to arrive at the same refusal."""
        fleet_calls: list[str] = []
        first = _FakeTrainer(initialized=True, trainer_id=_TRAINER_ID, fleet_calls=fleet_calls)
        second = _FakeTrainer(initialized=False, trainer_id=_OTHER_TRAINER_ID, fleet_calls=fleet_calls)

        with pytest.raises(AssertionError, match="disagree about being initialized"):
            await wait_trainers_idle({_TRAINER_ID: first, _OTHER_TRAINER_ID: second})

        assert "wait_idle" not in [call.split(".")[1] for call in fleet_calls]

    async def test_every_trainer_is_waited_out_on_the_same_budget(self):
        """Each wait is bounded on its own, so one slow trainer cannot eat what the next one gets."""
        first = _FakeTrainer(initialized=True)
        second = _FakeTrainer(initialized=True)

        await wait_trainers_idle({_TRAINER_ID: first, _OTHER_TRAINER_ID: second})

        assert first.idle_timeouts == [hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS]
        assert second.idle_timeouts == [hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS]

    async def test_a_take_over_that_reaches_no_trainer_at_all_is_refused(self):
        """A take-over with no trainer to wait out means the wiring never handed it the run's trainers."""
        with pytest.raises(AssertionError):
            await wait_trainers_idle({})

    async def test_a_trainer_that_never_goes_idle_fails_loud(self, short_take_over_budget: None):
        """A trainer that never finishes its call has to surface as the budget of its own wait running out."""
        trainer = _FakeTrainer(initialized=True, idle_seconds=_STALLED_SECONDS)

        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await wait_trainers_idle({_TRAINER_ID: trainer})

        assert time.monotonic() - started < _STALLED_SECONDS
        assert trainer.calls == ["is_initialized", "wait_idle"]

    async def test_trainers_that_disagree_about_being_initialized_stop_the_run(self):
        """A mixed fleet cannot be resumed, so it must stop here rather than half-build the run."""
        with pytest.raises(AssertionError, match="disagree about being initialized"):
            await wait_trainers_idle(
                {_TRAINER_ID: _FakeTrainer(initialized=True), _OTHER_TRAINER_ID: _FakeTrainer(initialized=False)}
            )


class TestTheTrainerStateIsRolledBack:
    async def test_a_cold_trainer_is_initialized_and_a_resumed_one_only_reloads(self):
        """Init rebuilds a trainer; a survivor must only be rolled back to its checkpoint."""
        cold = _FakeTrainer(initialized=False)
        warm = _FakeTrainer(initialized=True)

        assert await trainer_init_or_load_state(cold, Namespace(), trainer_id=_TRAINER_ID, resumed=False) == [7]
        assert await trainer_init_or_load_state(warm, Namespace(), trainer_id=_TRAINER_ID, resumed=True) == [3]
        assert cold.calls == ["init"] and warm.calls == ["load_state"]

    async def test_a_reload_that_never_returns_fails_loud(self, short_reload_budget: None):
        """A trainer wedged inside load_state would otherwise leave the run waiting on it forever."""
        trainer = _FakeTrainer(initialized=True, load_seconds=_STALLED_SECONDS)

        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await trainer_init_or_load_state(trainer, Namespace(), trainer_id=_TRAINER_ID, resumed=True)

        assert time.monotonic() - started < _STALLED_SECONDS

    async def test_each_trainer_gets_a_reload_budget_of_its_own(self, monkeypatch: pytest.MonkeyPatch):
        """A reload is bounded per trainer, so one slow policy cannot starve the policy reloaded after it."""
        monkeypatch.setattr(hot_restart_module, "_TRAINER_RELOAD_TIMEOUT_SECONDS", 0.3)
        first = _FakeTrainer(initialized=True, load_seconds=0.2)
        second = _FakeTrainer(initialized=True, load_seconds=0.2)

        assert await trainer_init_or_load_state(first, Namespace(), trainer_id=_TRAINER_ID, resumed=True) == [3]
        assert await trainer_init_or_load_state(second, Namespace(), trainer_id=_OTHER_TRAINER_ID, resumed=True) == [3]


class TestTheReloadHasABudgetOfItsOwn:
    def test_a_reload_is_given_far_longer_than_the_gate_that_precedes_it(self):
        """Loading a large checkpoint takes minutes, while the gate only waits for calls already running to end."""
        assert hot_restart_module._TRAINER_RELOAD_TIMEOUT_SECONDS > hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS

    async def test_a_reload_outlasting_the_gate_budget_is_not_cut_short(self, short_take_over_budget: None):
        """The gate budget is spent by the time the reload starts, and reusing it would refuse every real reload."""
        trainer = _FakeTrainer(initialized=True, load_seconds=_SHORT_BUDGET_SECONDS * 3)

        assert await trainer_init_or_load_state(trainer, Namespace(), trainer_id=_TRAINER_ID, resumed=True) == [3]

    async def test_a_reload_that_never_ends_fails_on_its_own_budget(self, short_reload_budget: None):
        """A reload nobody bounds would leave a hot restart hanging without ever starting training."""
        trainer = _FakeTrainer(initialized=True, load_seconds=_STALLED_SECONDS)

        with pytest.raises(asyncio.TimeoutError):
            await trainer_init_or_load_state(trainer, Namespace(), trainer_id=_TRAINER_ID, resumed=True)


class TestTheInferenceSideIsInitedOrReset:
    async def test_a_fresh_controller_is_simply_initialized(self):
        """A cold start must be untouched by the take-over protocol it shares this entry point with."""
        controller = _FakeInferenceController(initialized=False)

        await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert controller.calls == ["is_initialized", "init"]

    async def test_a_surviving_controller_is_reset_in_the_order_that_leaves_a_quiet_fleet(self):
        """A cell away during the abort would rejoin still generating, so the fleet is completed first."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert controller.calls == [
            "is_initialized",
            "wait_idle",
            "wait_expected_num_cells",
            "abort_all",
        ]

    async def test_a_surviving_controller_rejects_disk_delta_weight_transfer(self):
        """Disk-delta cannot adopt an engine because its first update only captures a baseline."""
        controller = _FakeInferenceController(initialized=True)

        with pytest.raises(AssertionError, match="does not support disk-delta weight transfer"):
            await init_or_reset_inference_controller(controller, args=_DISK_DELTA_ARGS)

        assert controller.calls == ["is_initialized"]

    async def test_a_call_of_the_previous_script_that_never_ends_fails_loud(self):
        """The script that died inside start_update_weights is exactly the case this wait exists for."""
        controller = _FakeInferenceController(initialized=True, busy=True)

        with pytest.raises(TimeoutError, match="still busy"):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert "abort_all" not in controller.calls

    async def test_each_step_after_the_drain_is_bounded_on_its_own(self):
        """A step that runs long must fail on its own timeout rather than eat what the next one gets."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert controller.fleet_timeouts == [hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS]

    async def test_the_drain_of_the_previous_script_gets_a_budget_of_its_own(self):
        """A cell being healed keeps a legitimate generation in flight for an hour, and that is not a gate step."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert controller.idle_timeouts == [hot_restart_module._INFERENCE_IDLE_TIMEOUT_SECONDS]
        assert hot_restart_module._INFERENCE_IDLE_TIMEOUT_SECONDS > hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS

    async def test_a_controller_that_never_answers_fails_loud(self, short_take_over_budget: None):
        """Hanging here would leave the operator with a silent hot restart that never starts training."""
        controller = _FakeInferenceController(initialized=True, wedged=True)

        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert time.monotonic() - started < _STALLED_SECONDS

    async def test_a_take_over_waits_for_the_whole_fleet_just_as_a_cold_start_does(self):
        """Generating on half a fleet because an engine was being rescheduled is not what the command asked for."""
        controller = _FakeInferenceController(initialized=True, fleet_incomplete=True)

        with pytest.raises(TimeoutError, match="short of engines"):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert "abort_all" not in controller.calls

    async def test_a_cell_that_refused_the_abort_fails_the_take_over(self):
        """The whole fleet was already there, so a refusal is a sick engine that may still be generating."""
        controller = _FakeInferenceController(
            initialized=True, abort_error=RuntimeError("west-engine-0-0-0 refused the abort")
        )

        with pytest.raises(RuntimeError, match="west-engine-0-0-0"):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)


class TestAbortInflightRollouts:
    async def test_a_refusing_cell_stops_the_run_instead_of_being_logged_past(self):
        """A cell that kept generating pollutes this run's data, so the take-over cannot continue over it."""
        controller = _FakeInferenceController(initialized=True, abort_error=RuntimeError("the cell refused"))

        with pytest.raises(RuntimeError, match="the cell refused"):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

    async def test_a_fleet_that_answered_every_abort_is_logged_as_asked_rather_than_as_quiet(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Nothing here confirms the generations stopped, so the line may only claim the aborts were accepted."""
        controller = _FakeInferenceController(initialized=True)

        with caplog.at_level(logging.INFO):
            await init_or_reset_inference_controller(controller, args=_BROADCAST_ARGS)

        assert "Asked every engine of the fleet to abort" in caplog.text


class _AbortingCell:
    def __init__(self, *, cell_id: str, failure: Exception | None = None) -> None:
        self.meta = SimpleNamespace(cell_id=cell_id, needs_offload=False, num_gpus_per_engine=1, gpu_offset=0)
        self.is_pending_weights_or_serving = True
        self.failure = failure
        self.aborted = False

    async def abort_all(self) -> None:
        self.aborted = True
        if self.failure is not None:
            raise self.failure


class _UnaddressedEngineProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"aborting a fleet never addresses a cell ({worker_name=})")


class TestEveryCellOfAServerIsAborted:
    @staticmethod
    def _server(cells: list[_AbortingCell]) -> RolloutServer:
        return RolloutServer(
            server_cells={cell.meta.cell_id: cell for cell in cells},
            args=SimpleNamespace(colocate=True),
            context_lock=ContextLock("InferenceController"),
            engine_provider=_UnaddressedEngineProvider(),
        )

    async def test_a_fleet_that_answered_every_abort_reports_no_refusal(self):
        """The ordinary take-over aborts every cell, and the gate above it has nothing to act on."""
        cells = [_AbortingCell(cell_id="west-0"), _AbortingCell(cell_id="west-1")]
        server = self._server(cells)

        async with server.context_lock:
            assert await server.abort_all() is None

        assert all(cell.aborted for cell in cells)

    async def test_every_refusing_cell_is_logged_before_the_run_stops(self, caplog: pytest.LogCaptureFixture):
        """An operator has to see every sick engine, not only the one the gather happened to order first."""
        cells = [
            _AbortingCell(cell_id="west-0", failure=RuntimeError("west-0 refused")),
            _AbortingCell(cell_id="west-1", failure=RuntimeError("west-1 refused")),
        ]
        server = self._server(cells)

        with caplog.at_level(logging.ERROR):
            async with server.context_lock:
                with pytest.raises(RuntimeError, match="west-0 refused"):
                    await server.abort_all()

        assert "west-0" in caplog.text and "west-1" in caplog.text
        assert all(cell.aborted for cell in cells)


class TestEveryServerOfTheFleetIsAborted:
    @staticmethod
    def _controller(cells: dict[str, _AbortingCell]) -> InferenceController:
        context_lock = ContextLock("InferenceController")
        controller = InferenceController.__new__(InferenceController)
        controller.context_lock = context_lock
        controller.servers = {
            model_name: RolloutServer(
                server_cells={cell.meta.cell_id: cell},
                args=SimpleNamespace(colocate=True),
                context_lock=context_lock,
                engine_provider=_UnaddressedEngineProvider(),
                model_name=model_name,
            )
            for model_name, cell in cells.items()
        }
        return controller

    async def test_a_fleet_whose_every_server_answered_reports_no_refusal(self):
        """One abort per model is the ordinary take-over, and the gate above it has nothing to act on."""
        cells = {"actor": _AbortingCell(cell_id="actor-0"), "ref": _AbortingCell(cell_id="ref-0")}

        assert await self._controller(cells).abort_all() is None

        assert all(cell.aborted for cell in cells.values())

    async def test_a_refusing_cell_of_every_server_is_logged_before_the_run_stops(
        self, caplog: pytest.LogCaptureFixture
    ):
        """One raising server must not hide the sick engines of the servers beside it, nor orphan their failures."""
        cells = {
            "actor": _AbortingCell(cell_id="actor-0", failure=RuntimeError("actor-0 refused")),
            "ref": _AbortingCell(cell_id="ref-0", failure=RuntimeError("ref-0 refused")),
        }

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="refused"):
                await self._controller(cells).abort_all()

        assert "actor-0" in caplog.text and "ref-0" in caplog.text
        assert all(cell.aborted for cell in cells.values())


class _WaitingServer:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    async def wait_init_expected_num_cells(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class TestTheWholeFleetIsWaitedForUnderTheCallersBudget:
    async def test_every_server_is_waited_for_under_the_budget_the_take_over_had_left(self):
        """One 3600s wait per model would blow the take-over budget the gate above promised to honour."""
        controller = InferenceController.__new__(InferenceController)
        controller.servers = {"actor": _WaitingServer(), "ref": _WaitingServer()}

        await controller.wait_expected_num_cells(timeout=12.5)

        assert [srv.timeouts for srv in controller.servers.values()] == [[12.5], [12.5]]


class _WireInferenceController:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release_previous_call = threading.Event()
        self.previous_call_started = threading.Event()

    async def is_initialized(self) -> bool:
        self.calls.append("is_initialized")
        return True

    def demo_previous_script_call(self) -> None:
        self.previous_call_started.set()
        assert self.release_previous_call.wait(timeout=30.0)
        self.calls.append("previous_script_call_end")

    async def wait_expected_num_cells(self, timeout: float) -> None:
        self.calls.append("wait_expected_num_cells")

    async def abort_all(self) -> None:
        self.calls.append("abort_all")


@contextlib.asynccontextmanager
async def _handle_onto_running_worker(worker: object, worker_cls: type) -> AsyncIterator[RpcWorkerHandle]:
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http_client:
            yield RpcWorkerHandle(worker_cls, server_url="http://testserver", http_client=http_client)


class TestTheTakeOverSurfaceCrossesTheWire:
    @pytest.mark.parametrize(
        "worker_cls, methods",
        [
            (TrainerController, {"is_initialized", "load_state"}),
            (RolloutExecutor, {"is_initialized"}),
            (InferenceController, {"is_initialized", "abort_all", "wait_expected_num_cells"}),
        ],
    )
    def test_the_take_over_surface_is_exposed_over_rpc(self, worker_cls: type, methods: set[str]):
        """A restarted orchestration script drives the whole take-over through exactly these rpc methods."""
        assert methods <= set(collect_rpc_method_specs(worker_cls))


class TestATakeOverDrivesARealInferenceControllerOverTheWire:
    async def test_the_whole_gate_runs_against_a_worker_a_previous_script_left_initialized(self):
        """Every step of this gate is an rpc call, so its real wire order is pinned against a real server."""
        worker = _WireInferenceController()

        async with _handle_onto_running_worker(worker, _WireInferenceController) as handle:
            await init_or_reset_inference_controller(handle, args=_BROADCAST_ARGS)

        assert worker.calls == ["is_initialized", "wait_expected_num_cells", "abort_all"]

    async def test_the_call_the_previous_script_left_running_is_waited_out_before_the_fleet_is_aborted(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Aborting while the previous script is still generating would leave its requests to finish after the reset."""
        monkeypatch.setattr(rpc_handle_module, "_IDLE_POLL_INTERVAL_SECONDS", 0.02)
        worker = _WireInferenceController()

        async def _release_soon() -> None:
            await asyncio.sleep(0.2)
            worker.release_previous_call.set()

        async with _handle_onto_running_worker(worker, _WireInferenceController) as handle:
            previous_call = asyncio.create_task(handle.demo_previous_script_call())
            assert await asyncio.to_thread(worker.previous_call_started.wait, 5.0)
            releaser = asyncio.create_task(_release_soon())

            await init_or_reset_inference_controller(handle, args=_BROADCAST_ARGS)

            await previous_call
            await releaser

        assert worker.calls.index("previous_script_call_end") < worker.calls.index("abort_all")
