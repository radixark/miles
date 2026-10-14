import asyncio
import threading

import pytest
from tests.fast.utils.workers.rpc.server.lifecycle_fixtures import _run_process_probe
from tests.fast.utils.workers.rpc.server.worker_loop_fixtures import (
    _blocked_checkpoint_probe,
    _worker_loop_failure_probe,
)

from miles.utils.workers.rpc.server.worker_loop import _DedicatedWorkerLoop


class TestWorkerLoopLifecycle:
    async def test_cancelling_a_call_cancels_work_on_the_worker_loop(self) -> None:
        """Cancelling the HTTP-side task must reach the isolated worker coroutine."""
        started = threading.Event()
        cancelled = threading.Event()
        worker_loop = _DedicatedWorkerLoop()

        async def block() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(worker_loop.call(method=block, kwargs={}, is_async=True))
        try:
            assert await asyncio.to_thread(started.wait, 2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(cancelled.wait, 2.0)
        finally:
            await worker_loop.close()
        assert not worker_loop._thread.is_alive()

    async def test_exceptions_reach_the_caller_and_close_rejects_new_work(self) -> None:
        """Isolation must preserve worker failures and prevent submissions after shutdown."""
        worker_loop = _DedicatedWorkerLoop()

        async def fail() -> None:
            raise ValueError("worker failure")

        try:
            with pytest.raises(ValueError, match="worker failure"):
                await worker_loop.call(method=fail, kwargs={}, is_async=True)
        finally:
            await worker_loop.close()
        await worker_loop.close()
        with pytest.raises(RuntimeError, match="closed"):
            await worker_loop.call(method=fail, kwargs={}, is_async=True)


class TestWorkerLoopProcessFailure:
    @pytest.mark.parametrize("mode", ["system_exit", "keyboard_interrupt", "stop", "shutdown"])
    def test_a_dead_worker_loop_cannot_leave_a_healthy_process(self, mode: str) -> None:
        """A fatal worker loop exit must terminate its owning process."""
        assert _run_process_probe(target=_worker_loop_failure_probe, mode=mode) == 1

    def test_blocked_checkpoint_cannot_prevent_process_shutdown(self) -> None:
        """The shutdown deadline also covers synchronous I/O on the dedicated loop."""
        assert _run_process_probe(target=_blocked_checkpoint_probe, mode="checkpoint") == 1
