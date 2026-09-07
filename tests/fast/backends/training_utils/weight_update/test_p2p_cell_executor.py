import threading
import time

import pytest

from miles.backends.training_utils.weight_update.protocols.p2p_cell_executor import _CellWriteExecutor


def _blocking_task(started: threading.Event, release: threading.Event):
    def run() -> None:
        started.set()
        assert release.wait(timeout=30.0)

    return run


class TestPerCellIsolation:
    """A write thread belongs to one inference cell and can only ever hold up that cell."""

    def test_a_blocked_cell_does_not_hold_up_another_one(self) -> None:
        """A shared pool of four workers is exhausted by four stuck cells, which stalls every healthy one."""
        blocked = _CellWriteExecutor("cell-blocked")
        healthy = _CellWriteExecutor("cell-healthy")
        started, release = threading.Event(), threading.Event()

        try:
            blocked.submit(_blocking_task(started, release))
            assert started.wait(timeout=30.0)

            assert healthy.submit(lambda: "written").result(timeout=30.0) == "written"
        finally:
            release.set()

    def test_the_writes_of_one_cell_run_in_order(self) -> None:
        """Two writes of one cell read the same shared buffers, so overlapping them would race over the source."""
        executor = _CellWriteExecutor("cell-0")
        running = threading.Event()
        observed: list[bool] = []

        def first() -> None:
            running.set()
            observed.append(True)

        def second() -> None:
            observed.append(running.is_set())

        executor.submit(first)
        executor.submit(second).result(timeout=30.0)

        assert observed == [True, True]

    def test_a_failing_write_surfaces_through_its_own_future(self) -> None:
        """The submitting cell updater is the only place that knows whose failure this is."""
        executor = _CellWriteExecutor("cell-0")

        def failing() -> None:
            raise RuntimeError("[P2P-Shared] Transfer failed for session s0, error: -1")

        future = executor.submit(failing)

        assert isinstance(future.exception(timeout=30.0), RuntimeError)

    def test_a_cell_that_never_wrote_starts_no_thread(self) -> None:
        """Every rank tracks every cell, and most of them hold no target of it at all."""
        executor = _CellWriteExecutor("cell-0")

        assert executor.is_running is False
        assert executor.close() is True


class TestDisposal:
    """Releasing a cell must not wait on a native write that may never come back."""

    def test_an_idle_executor_is_released(self) -> None:
        """A thread per reconnection per cell would grow without bound over a long run."""
        executor = _CellWriteExecutor("cell-0")
        executor.submit(lambda: None).result(timeout=30.0)

        assert executor.close(timeout=30.0) is True
        assert executor.is_running is False

    def test_a_blocked_executor_is_reported_instead_of_joined(self) -> None:
        """Joining a thread stuck in a native transfer would hang the actor teardown forever."""
        executor = _CellWriteExecutor("cell-0")
        started, release = threading.Event(), threading.Event()

        try:
            executor.submit(_blocking_task(started, release))
            assert started.wait(timeout=30.0)

            assert executor.close() is False
            assert executor.is_running is True
        finally:
            release.set()

    def test_a_closed_executor_refuses_new_writes(self) -> None:
        """A write queued behind a closed executor would never run and never be collected."""
        executor = _CellWriteExecutor("cell-0")
        executor.close()

        with pytest.raises(AssertionError, match="already closed"):
            executor.submit(lambda: None)

    def test_the_worker_thread_is_a_daemon(self) -> None:
        """A non-daemon thread stuck in a native transfer keeps the interpreter from exiting."""
        executor = _CellWriteExecutor("cell-0")
        started, release = threading.Event(), threading.Event()

        try:
            executor.submit(_blocking_task(started, release))
            assert started.wait(timeout=30.0)

            assert executor._thread.daemon is True
        finally:
            release.set()


class TestShutdownGrace:
    """A thread that only needs a moment to notice the sentinel is not a stuck one."""

    def test_a_thread_between_writes_is_released_rather_than_reported(self) -> None:
        """Reporting it stalled would count a scheduling gap against the bound that retires the trainer rank."""
        executor = _CellWriteExecutor("cell-0")
        executor.submit(lambda: None).result(timeout=30.0)

        assert executor.close() is True
        assert executor.is_running is False

    def test_the_close_of_a_stuck_thread_is_bounded(self) -> None:
        """A native transfer can outlive the actor, so teardown must give up on it rather than join it."""
        executor = _CellWriteExecutor("cell-0")
        started, release = threading.Event(), threading.Event()

        try:
            executor.submit(_blocking_task(started, release))
            assert started.wait(timeout=30.0)

            started_at = time.monotonic()
            assert executor.close(timeout=0.05) is False
            assert time.monotonic() - started_at < 5.0
        finally:
            release.set()
