import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import TrainerCpuWitnessEvent, TrainerWitnessCohortEvent
from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


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
        payloads: list[dict[str, Any]] = []

        class Controller:
            async def is_cpu_witness_snapshot_busy(self) -> bool:
                return False

            async def log_current_cpu_witness(self, *, rollout_id: int) -> dict[str, Any]:
                calls.append(f"witness:{rollout_id}")
                return {"snapshots": [], "marker": {"cohort": len(calls)}}

        def analyze(_args, *, process_started_at: datetime) -> None:
            assert process_started_at.tzinfo is timezone.utc
            calls.append("analyze")
            if calls.count("analyze") == 2:
                raise ValueError("periodic failure")

        monkeypatch.setattr(
            rollout_executor_module.event_analyzer,
            "run_sample_ownership_analysis_from_args",
            analyze,
        )
        monkeypatch.setattr(RolloutExecutor, "_log_current_cpu_witness", staticmethod(payloads.append))
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            sample_ownership_check_interval_seconds=0.0,
            sample_ownership_check_timeout_seconds=1.0,
            sample_ownership_grace_period_seconds=300.0,
        )
        executor.rollout_id = 7
        executor._actor_controller = Controller()
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_busy_since = None

        with pytest.raises(ValueError, match="periodic failure"):
            await executor._run_sample_ownership_checker()

        assert calls == ["witness:7", "analyze", "witness:7", "analyze"]
        assert payloads == [
            {"snapshots": [], "marker": {"cohort": 1}},
            {"snapshots": [], "marker": {"cohort": 3}},
        ]

    async def test_witness_rpc_timeout_is_fatal(self) -> None:
        """A bounded witness refresh cannot leave the checker silently hung forever."""

        class Controller:
            async def is_cpu_witness_snapshot_busy(self) -> bool:
                return False

            async def log_current_cpu_witness(self, *, rollout_id: int) -> str:
                await asyncio.Event().wait()
                return "unreachable"

        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            sample_ownership_check_interval_seconds=0.0,
            sample_ownership_check_timeout_seconds=0.001,
            sample_ownership_grace_period_seconds=300.0,
        )
        executor.rollout_id = 7
        executor._actor_controller = Controller()
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_busy_since = None

        with pytest.raises(TimeoutError):
            await executor._run_sample_ownership_checker()


class TestCurrentWitnessCollection:
    def test_remote_snapshots_are_written_before_their_completion_marker(self, tmp_path: Path) -> None:
        """The local accounting stream receives complete remote evidence in commit order."""
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        payload = {
            "snapshots": [
                {
                    "type": "trainer_cpu_witness",
                    "timestamp": timestamp.isoformat(),
                    "source": TrainProcessIdentity(
                        component="actor",
                        cell_index=0,
                        rank_within_cell=0,
                    ).model_dump(mode="json"),
                    "replica_id": "cell-0",
                    "rollout_id": 4,
                    "cohort_id": "cohort-4",
                    "sample_counts": [],
                    "reason": "current",
                }
            ],
            "marker": {
                "type": "trainer_witness_cohort",
                "timestamp": timestamp.isoformat(),
                "source": TrainerControllerProcessIdentity(trainer_id="actor").model_dump(mode="json"),
                "rollout_id": 4,
                "cohort_id": "cohort-4",
                "replica_ids": ["cell-0"],
            },
        }
        event_logger = EventLogger(
            log_dir=tmp_path,
            source=SimpleProcessIdentity(component="rollout_executor"),
        )
        set_event_logger(event_logger)
        try:
            RolloutExecutor._log_current_cpu_witness(payload)
        finally:
            set_event_logger(None)

        events = read_events(tmp_path)
        assert [type(event) for event in events] == [TrainerCpuWitnessEvent, TrainerWitnessCohortEvent]
        assert events[0].source.component == "actor"
        assert events[1].source.component == "trainer_controller"


class TestBusyTrainerSampleOwnershipCheck:
    async def test_first_busy_training_waits_for_a_completed_cohort(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first long optimizer mutation gets its stall bound before current-cohort analysis begins."""

        class Controller:
            async def is_cpu_witness_snapshot_busy(self) -> bool:
                return True

        monkeypatch.setattr(
            rollout_executor_module.event_analyzer,
            "run_sample_ownership_analysis_from_args",
            lambda *_args, **_kwargs: pytest.fail("analysis requires a completed witness cohort"),
        )
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            sample_ownership_check_timeout_seconds=1.0,
            sample_ownership_grace_period_seconds=300.0,
        )
        executor.rollout_id = 7
        executor._actor_controller = Controller()
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_busy_since = None
        executor._sample_ownership_has_cohort = False

        await executor._run_one_sample_ownership_check()

        assert executor._sample_ownership_busy_since is not None

    async def test_busy_training_uses_existing_evidence_until_the_stall_bound(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A normal in-flight optimizer mutation does not block periodic analysis or trigger a snapshot race."""
        analyzed: list[datetime] = []

        class Controller:
            async def is_cpu_witness_snapshot_busy(self) -> bool:
                return True

            async def log_current_cpu_witness(self, *, rollout_id: int) -> str:
                raise AssertionError("a busy trainer must not be snapshotted")

        def analyze(_args, *, process_started_at: datetime) -> None:
            analyzed.append(process_started_at)

        monkeypatch.setattr(
            rollout_executor_module.event_analyzer,
            "run_sample_ownership_analysis_from_args",
            analyze,
        )
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            sample_ownership_check_timeout_seconds=1.0,
            sample_ownership_grace_period_seconds=300.0,
        )
        executor.rollout_id = 7
        executor._actor_controller = Controller()
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_busy_since = None
        executor._sample_ownership_has_cohort = True

        await executor._run_one_sample_ownership_check()
        await executor._run_one_sample_ownership_check()

        assert len(analyzed) == 2

    async def test_a_continuously_busy_trainer_eventually_fails_the_main_checker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Busy state cannot suppress fresh current-weight evidence forever."""

        class Controller:
            async def is_cpu_witness_snapshot_busy(self) -> bool:
                return True

        monkeypatch.setattr(
            rollout_executor_module.event_analyzer,
            "run_sample_ownership_analysis_from_args",
            lambda _args, *, process_started_at: None,
        )
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            sample_ownership_check_timeout_seconds=1.0,
            sample_ownership_grace_period_seconds=30.0,
        )
        executor.rollout_id = 7
        executor._actor_controller = Controller()
        executor._sample_ownership_started_at = datetime.now(timezone.utc)
        executor._sample_ownership_busy_since = rollout_executor_module.time.monotonic() - 60.0
        executor._sample_ownership_has_cohort = False

        with pytest.raises(TimeoutError, match="remained busy"):
            await executor._run_one_sample_ownership_check()


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
