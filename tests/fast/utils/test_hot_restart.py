from __future__ import annotations

import asyncio
import time
from argparse import Namespace
from typing import Any

import httpx
import pytest

from miles.utils import hot_restart as hot_restart_module
from miles.utils.hot_restart import trainer_init_or_load_state, wait_trainers_idle, wait_until_worker_not_initialized
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError

_TRAINER_ID = "policy_a-actor"
_OTHER_TRAINER_ID = "policy_b-actor"
_STALLED_SECONDS = 5.0
_SHORT_BUDGET_SECONDS = 0.05


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
