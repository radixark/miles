import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.audit_utils.sample_ownership import checker as checker_module


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
    async def test_non_ci_errors_are_logged_and_the_checker_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        """Ordinary runs report failures while keeping future checks alive."""
        attempts = 0

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ValueError("missing training outcome")
                raise asyncio.CancelledError

        executor = self._executor(controller=Controller(), store=SimpleNamespace(), interval=0)
        executor.args.ci_test = False

        with pytest.raises(asyncio.CancelledError):
            await executor._run_sample_ownership_checker()

        assert attempts == 2
        assert any(record.levelname == "ERROR" and record.exc_info for record in caplog.records)
        assert "missing training outcome" in caplog.text

    async def test_checks_continue_without_any_rollout_completion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The periodic loop refreshes witnesses and analyzes events without depending on get completion."""
        calls: list[str] = []

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                calls.append(f"witness:{rollout_id}")
                return {"snapshots": [], "marker": {"cohort_id": str(len(calls))}}

        class Store:
            def replace_current(self, payload: dict[str, Any]) -> SimpleNamespace:
                calls.append(f"replace:{payload['marker']['cohort_id']}")
                return SimpleNamespace(marker=SimpleNamespace(mature_before=datetime.now(timezone.utc)))

            def read_events(self) -> list[Any]:
                calls.append("read")
                return []

        def analyze(_events: list[Any], **_kwargs: Any) -> None:
            calls.append("analyze")
            if calls.count("analyze") == 2:
                raise ValueError("periodic failure")

        monkeypatch.setattr(checker_module, "run_sample_ownership_analysis", analyze)
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

    async def test_the_completed_step_window_sets_the_analysis_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only samples older than the trainer's completed step window are checked."""
        request_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        observed: list[datetime] = []

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                return {"snapshots": [], "marker": {}}

        class Store:
            def replace_current(self, payload: dict[str, Any]) -> SimpleNamespace:
                return SimpleNamespace(marker=SimpleNamespace(mature_before=request_time))

            def read_events(self) -> list[Any]:
                return []

        def analyze(_events: list[Any], **kwargs: Any) -> None:
            observed.append(kwargs["now"])

        monkeypatch.setattr(checker_module, "run_sample_ownership_analysis", analyze)
        executor = self._executor(controller=Controller(), store=Store())

        await executor._run_one_sample_ownership_check()

        assert observed == [request_time]

    async def test_queued_snapshot_timeout_is_fatal(self) -> None:
        """A trainer lock wait cannot leave the ownership checker silently hung forever."""

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> None:
                await asyncio.Event().wait()

        executor = self._executor(controller=Controller(), store=SimpleNamespace(), timeout=0.001)

        with pytest.raises(TimeoutError):
            await executor._run_one_sample_ownership_check()

    async def test_current_store_timeout_is_fatal(self) -> None:
        """A blocked current-cohort replacement cannot stop checker error propagation."""

        class Controller:
            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                return {"snapshots": [], "marker": {}}

        class Store:
            def replace_current(self, payload: dict[str, Any]) -> None:
                time.sleep(1)

        executor = self._executor(controller=Controller(), store=Store(), timeout=0.001)

        with pytest.raises(TimeoutError):
            await executor._run_one_sample_ownership_check()

    @staticmethod
    def _executor(
        *,
        controller: Any,
        store: Any,
        interval: float = 30.0,
        timeout: float = 1.0,
    ) -> RolloutExecutor:
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            save_debug_event_data="/events",
            sample_ownership_check_interval_seconds=interval,
            sample_ownership_check_timeout_seconds=timeout,
            ci_test=True,
        )
        executor.rollout_id = 7
        executor._actor_controller = controller
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
