import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.workers.worker_handle import WorkerStillBusyError


class TestRolloutWaitsForSampleOwnership:
    async def test_a_checker_failure_interrupts_a_rollout_waiting_for_data(self) -> None:
        """A missing batch cannot hide a checker failure from the orchestration loop."""
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_rollout(**_kwargs: Any) -> None:
            try:
                await release.wait()
            finally:
                cancelled.set()

        async def failing_check() -> None:
            await asyncio.sleep(0)
            raise ValueError("sample ownership failed")

        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor._get_rollout_data = blocked_rollout
        executor._sample_ownership_task = asyncio.create_task(failing_check())

        with pytest.raises(ValueError, match="sample ownership failed"):
            await executor._get_rollout_data_with_ownership_check(rollout_id=3, trainer_model_id=None)

        assert cancelled.is_set()

    async def test_a_finished_rollout_does_not_stop_future_checks(self) -> None:
        """Completing one get leaves the independent periodic checker running for later samples."""
        checker_release = asyncio.Event()

        async def rollout(**_kwargs: Any) -> tuple[str, str, str]:
            return "data", "metadata", "metrics"

        async def checker() -> None:
            await checker_release.wait()

        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor._get_rollout_data = rollout
        executor._sample_ownership_task = asyncio.create_task(checker())

        assert await executor._get_rollout_data_with_ownership_check(
            rollout_id=3,
            trainer_model_id=None,
        ) == ("data", "metadata", "metrics")
        assert not executor._sample_ownership_task.done()

        checker_release.set()
        await executor._sample_ownership_task


class TestPeriodicSampleOwnershipCheck:
    async def test_checks_continue_without_any_rollout_completion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The periodic loop refreshes witnesses and analyzes events without depending on get completion."""
        calls: list[str] = []

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                calls.append(f"witness:{rollout_id}")
                return {"snapshots": [], "marker": {"cohort_id": str(len(calls))}}

        class Store:
            def replace_current(self, payload: dict[str, Any]) -> None:
                calls.append(f"replace:{payload['marker']['cohort_id']}")

            def read_events(self) -> list[Any]:
                calls.append("read")
                return []

        def analyze(_events: list[Any], **_kwargs: Any) -> None:
            calls.append("analyze")
            if calls.count("analyze") == 2:
                raise ValueError("periodic failure")

        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_sample_ownership_analysis", analyze)
        executor = self._executor(controller=Controller(), store=Store(), interval=0.0)

        with pytest.raises(ValueError, match="periodic failure"):
            await executor._run_sample_ownership_checker()

        assert calls == [
            "witness:7",
            "replace:1",
            "read",
            "analyze",
            "witness:7",
            "replace:5",
            "read",
            "analyze",
        ]

    async def test_snapshot_request_time_is_the_analysis_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Samples maturing while a queued snapshot waits are checked against the request-time weights."""
        request_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        response_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
        clock = iter((request_time, response_time))
        observed: list[datetime] = []

        class Clock:
            @classmethod
            def now(cls, tz: timezone) -> datetime:
                assert tz is timezone.utc
                return next(clock)

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                Clock.now(timezone.utc)
                return {"snapshots": [], "marker": {}}

        class Store:
            def replace_current(self, payload: dict[str, Any]) -> None:
                return None

            def read_events(self) -> list[Any]:
                return []

        def analyze(_events: list[Any], **kwargs: Any) -> None:
            observed.append(kwargs["now"])

        monkeypatch.setattr(rollout_executor_module, "datetime", Clock)
        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_sample_ownership_analysis", analyze)
        executor = self._executor(controller=Controller(), store=Store())

        await executor._run_one_sample_ownership_check()

        assert observed == [request_time]

    async def test_queued_snapshot_timeout_is_fatal(self) -> None:
        """A trainer lock wait cannot leave the ownership checker silently hung forever."""

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> None:
                await asyncio.Event().wait()

        executor = self._executor(controller=Controller(), store=SimpleNamespace(), grace=0.001, timeout=0.001)

        with pytest.raises(TimeoutError):
            await executor._run_one_sample_ownership_check()

    async def test_transient_cohort_changes_retry_within_one_deadline(self) -> None:
        """A fleet transition retries fresh collection without resetting the overall timeout."""
        calls = 0

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise WorkerStillBusyError("trainer cell cohort changed")
                return {"snapshots": [], "marker": {}}

        executor = self._executor(controller=Controller(), store=SimpleNamespace(), interval=0.0)

        payload, _ = await executor._collect_current_cpu_witness(timeout=1.0)

        assert payload == {"snapshots": [], "marker": {}}
        assert calls == 2

    @staticmethod
    def _executor(
        *,
        controller: Any,
        store: Any,
        interval: float = 30.0,
        grace: float = 300.0,
        timeout: float = 1.0,
    ) -> RolloutExecutor:
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            save_debug_event_data="/events",
            sample_ownership_check_interval_seconds=interval,
            sample_ownership_check_timeout_seconds=timeout,
            sample_ownership_grace_period_seconds=grace,
        )
        executor.rollout_id = 7
        executor._actor_controller = controller
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_store = store
        return executor


class TestStopSampleOwnershipChecker:
    async def test_dispose_recovers_an_already_failed_checker_error(self) -> None:
        """A failure that happened between gets remains observable during lifecycle cleanup."""

        async def failing_check() -> None:
            raise RuntimeError("checker died")

        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor._sample_ownership_task = asyncio.create_task(failing_check())
        await asyncio.sleep(0)

        error = await executor._stop_sample_ownership_checker()

        assert isinstance(error, RuntimeError)
        assert str(error) == "checker died"
