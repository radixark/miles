import threading

import pytest

from miles.utils.test_utils.polling_worker import PollingWorker, poll_until_stopped

_JOIN_TIMEOUT_SECONDS = 5.0
_WORKER_NAME = "test-polling-worker"


def _find_thread(name: str) -> threading.Thread | None:
    return next((thread for thread in threading.enumerate() if thread.name == name), None)


class TestPollingWorker:
    def test_the_target_runs_on_a_daemon_thread_driven_by_the_stop_event(self) -> None:
        """The worker must not keep the process alive, and its target needs the event to notice a stop."""
        received: list[threading.Event] = []
        started = threading.Event()

        def run(stop_event: threading.Event) -> None:
            received.append(stop_event)
            started.set()
            stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)

        worker = PollingWorker(name=_WORKER_NAME, run=run)
        worker.start()

        assert started.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        assert worker.is_running
        thread = _find_thread(_WORKER_NAME)
        assert thread is not None and thread.daemon

        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        assert not worker.is_running
        assert len(received) == 1 and received[0].is_set()

    def test_stop_and_join_returns_with_the_worker_gone_within_the_timeout(self) -> None:
        """A worker still alive after the bounded join would race whatever reads its log next."""
        worker = PollingWorker(
            name=_WORKER_NAME, run=lambda stop_event: stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        )
        worker.start()

        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        assert not worker.is_running

    def test_assert_not_running_raises_the_given_message_while_the_thread_is_alive(self) -> None:
        """The caller's message names which worker outlived its join, so it must reach the failure."""
        release = threading.Event()
        worker = PollingWorker(name=_WORKER_NAME, run=lambda stop_event: release.wait(timeout=_JOIN_TIMEOUT_SECONDS))
        worker.start()

        try:
            with pytest.raises(AssertionError, match="injector still running"):
                worker.assert_not_running(message="injector still running")
        finally:
            release.set()
            worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

    def test_assert_not_running_passes_after_the_worker_stopped(self) -> None:
        """A worker that honoured its stop event must pass the check its callers gate the log read on."""
        worker = PollingWorker(
            name=_WORKER_NAME, run=lambda stop_event: stop_event.wait(timeout=_JOIN_TIMEOUT_SECONDS)
        )
        worker.start()
        worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)

        worker.assert_not_running(message="worker should be gone")

    def test_a_target_exception_is_rethrown_by_the_joining_caller(self) -> None:
        """A daemon target failure must fail the scenario instead of disappearing on its worker thread."""
        worker = PollingWorker(
            name=_WORKER_NAME,
            run=lambda stop_event: (_ for _ in ()).throw(KeyError("missing generation")),
        )
        worker.start()

        with pytest.raises(KeyError, match="missing generation"):
            worker.stop_and_join(timeout_seconds=_JOIN_TIMEOUT_SECONDS)


class TestPollUntilStopped:
    def test_it_keeps_ticking_until_the_stop_event_is_set(self) -> None:
        """The loop is what drives the periodic work, so it must not stop before the event is set."""
        stop_event = threading.Event()
        num_ticks = 0

        def tick() -> None:
            nonlocal num_ticks
            num_ticks += 1
            if num_ticks == 3:
                stop_event.set()

        poll_until_stopped(stop_event, tick=tick, poll_interval_seconds=0.0)

        assert num_ticks == 3

    def test_an_already_stopped_event_prevents_any_tick(self) -> None:
        """A worker stopped before its loop began must not inject anything the caller no longer expects."""
        stop_event = threading.Event()
        stop_event.set()
        ticks: list[None] = []

        poll_until_stopped(stop_event, tick=lambda: ticks.append(None), poll_interval_seconds=0.0)

        assert ticks == []

    def test_the_wait_between_ticks_uses_the_configured_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hard-coded sleep would ignore the caller's pacing and make the loop unstoppable for that long."""
        stop_event = threading.Event()
        waited: list[float | None] = []

        def recording_wait(timeout: float | None = None) -> bool:
            waited.append(timeout)
            stop_event.set()
            return True

        monkeypatch.setattr(stop_event, "wait", recording_wait)

        poll_until_stopped(stop_event, tick=lambda: None, poll_interval_seconds=0.25)

        assert waited == [0.25]
