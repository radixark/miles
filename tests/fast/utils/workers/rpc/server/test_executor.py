import asyncio
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
from tests.fast.utils.workers.rpc.server.fake_workers import (
    AsyncOnlyWorker,
    ExecutorUnderTest,
    OutcomeRecorder,
    SyncAndAsyncWorker,
)

from miles.utils.workers.rpc.common.metadata import rpc


class _OversizedWorker:
    @rpc(max_serialized_outcome_bytes=512)
    def demo_oversized(self) -> str:
        return "x" * 1024


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


class TestOutcomeSize:
    async def test_oversized_result_becomes_an_explicit_bounded_protocol_failure(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        """A method violating its declared result bound returns a small explicit RPC failure."""
        under_test = make_executor(_OversizedWorker)
        recorder = OutcomeRecorder()

        under_test.executor.start(
            spec=under_test.specs["demo_oversized"], kwargs={}, call_id="c1", finish=recorder.finish
        )
        await asyncio.gather(*under_test.executor._background_tasks)

        assert len(recorder.outcomes) == 1
        assert recorder.outcomes[0].status == "failed"
        assert "RpcOutcomeTooLargeError" in recorder.outcomes[0].error
        assert len(recorder.outcomes[0].model_dump_json().encode()) <= 512

    @pytest.mark.parametrize("mode", ["result", "error"])
    def test_large_result_and_traceback_encoding_have_a_bounded_peak_rss(self, mode: str) -> None:
        """Result and traceback size enforcement does not materialize a second unbounded wire copy."""
        script = textwrap.dedent(
            f"""
            import asyncio
            import json
            import resource

            from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, rpc
            from miles.utils.workers.rpc.server.executor import RpcCallExecutor

            class Worker:
                @rpc(max_serialized_outcome_bytes=512)
                def run(self) -> str:
                    payload = 'x' * (64 * 1024 * 1024)
                    if {mode!r} == 'error':
                        raise RuntimeError(payload)
                    return payload

            async def main():
                baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                specs = collect_rpc_method_specs(Worker)
                outcomes = []
                executor = RpcCallExecutor(worker=Worker(), specs=specs)
                executor.start(
                    spec=specs['run'],
                    kwargs={{}},
                    call_id='c1',
                    finish=lambda *, outcome: outcomes.append(outcome),
                )
                await asyncio.gather(*executor._background_tasks)
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                print(json.dumps({{
                    'rss_delta': peak - baseline,
                    'status': outcomes[0].status,
                    'outcome_bytes': len(outcomes[0].model_dump_json().encode()),
                }}))

            asyncio.run(main())
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        metrics = json.loads(completed.stdout.splitlines()[-1])

        assert metrics["status"] == "failed"
        assert metrics["outcome_bytes"] <= 512
        assert metrics["rss_delta"] <= 96 * 1024 * 1024
