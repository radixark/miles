"""Tests for the stop_clear_submit recovery primitive."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator.helpers import stop_clear_submit
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import FakeTrainingJob


class TestStopClearSubmit:
    @pytest.mark.anyio
    async def test_happy_path_returns_true(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 1.0})

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is True
        assert training_job._stopped
        assert training_job._submitted
        assert mini_wandb.latest(metric_name="loss") is None

    @pytest.mark.anyio
    async def test_stop_failure_but_job_stopped_still_submits(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.STOPPED])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is True
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_job_still_running_skips_submit(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is False
        assert not training_job._submitted

    @pytest.mark.anyio
    async def test_submit_failure_returns_false(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is False
        assert training_job._stopped

    @pytest.mark.anyio
    async def test_excluded_node_ids_passed_to_submit(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        result = await stop_clear_submit(
            training_job, mini_wandb, excluded_node_ids=["node-x", "node-y"],
        )

        assert result is True
        assert training_job._last_excluded_node_ids == ["node-x", "node-y"]

    @pytest.mark.anyio
    async def test_clear_happens_after_successful_submit(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 2.0})

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is True
        assert mini_wandb.latest(metric_name="loss") is None

    @pytest.mark.anyio
    async def test_clear_not_called_on_submit_failure(self) -> None:
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()
        mini_wandb.log_step(run_id="r", step=1, metrics={"loss": 3.0})

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        training_job.submit_training = failing_submit  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, mini_wandb)

        assert result is False
        assert mini_wandb.latest(metric_name="loss") == 3.0

    @pytest.mark.anyio
    async def test_stop_failure_job_failed_still_submits(self) -> None:
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")

        training_job.stop_training = failing_stop  # type: ignore[assignment]

        result = await stop_clear_submit(training_job, MiniWandb())

        assert result is True
        assert training_job._submitted
