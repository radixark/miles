from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from miles.utils import hot_restart as hot_restart_module
from miles.utils.hot_restart import wait_until_worker_not_initialized
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError

_SHORT_BUDGET_SECONDS = 0.05


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
