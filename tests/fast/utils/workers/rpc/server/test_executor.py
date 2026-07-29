import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from tests.fast.utils.workers.rpc.server.fake_workers import (
    AsyncOnlyWorker,
    ExecutorUnderTest,
    OutcomeRecorder,
    SyncAndAsyncWorker,
)


class TestConcurrencyGroups:
    def test_concurrency_groups_reports_unique_sorted_sync_groups(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        """Each sync concurrency group is reported exactly once, in sorted order."""
        assert make_executor(SyncAndAsyncWorker).executor.concurrency_groups == ["default", "left"]

    def test_async_only_worker_gets_no_concurrency_group(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        """A worker with only async methods needs no executor group at all."""
        assert make_executor(AsyncOnlyWorker).executor.concurrency_groups == []


class TestBackgroundTasks:
    @pytest.mark.parametrize(
        ("method_name", "kwargs", "status"),
        [("demo_sync", {"value": 1}, "success"), ("demo_sync_raises", {}, "failed")],
    )
    async def test_finished_tasks_are_removed_from_background_task_set(
        self,
        make_executor: Callable[[type], ExecutorUnderTest],
        method_name: str,
        kwargs: dict[str, Any],
        status: str,
    ) -> None:
        """A started call is held while it runs and dropped once it reaches a terminal outcome."""
        under_test = make_executor(SyncAndAsyncWorker)
        recorder = OutcomeRecorder()

        under_test.executor.start(
            spec=under_test.specs[method_name], kwargs=kwargs, call_id="c1", finish=recorder.finish
        )
        running = set(under_test.executor._background_tasks)
        assert len(running) == 1

        await asyncio.gather(*running)

        assert [outcome.status for outcome in recorder.outcomes] == [status]
        assert under_test.executor._background_tasks == set()

    async def test_cancelled_tasks_are_removed_from_background_task_set(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        """A task cancelled while its worker is running is dropped after cancellation finishes."""
        under_test = make_executor(AsyncOnlyWorker)
        worker = under_test.executor._worker
        assert isinstance(worker, AsyncOnlyWorker)
        recorder = OutcomeRecorder()

        under_test.executor.start(
            spec=under_test.specs["demo_async_blocks"], kwargs={}, call_id="c1", finish=recorder.finish
        )
        task = next(iter(under_test.executor._background_tasks))
        await asyncio.wait_for(worker.started.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        assert [outcome.status for outcome in recorder.outcomes] == ["failed"]
        assert under_test.executor._background_tasks == set()
