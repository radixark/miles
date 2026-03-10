"""Integration tests: RayTrainingJob submit and status lifecycle against a real local Ray cluster."""

from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.adapters.impl.ray.training_job import RayTrainingJob
from miles.utils.ft.adapters.types import JobStatus
from tests.fast.utils.ft.integration.local_ray_training_job.conftest import poll_until_terminal

pytestmark = [
    pytest.mark.local_ray,
]


class TestSubmitAndStatus:
    async def test_submit_returns_8char_run_id(
        self,
        make_training_job: ...,
    ) -> None:
        job: RayTrainingJob = make_training_job()
        run_id = await job.submit_training()

        assert len(run_id) == 8
        assert run_id.isalnum()

    async def test_quick_job_reaches_stopped_status(
        self,
        make_training_job: ...,
    ) -> None:
        """A trivial 'print(42)' job should reach STOPPED (SUCCEEDED mapped to STOPPED)."""
        job: RayTrainingJob = make_training_job(entrypoint='python -c "print(42)"')
        await job.submit_training()

        status = await poll_until_terminal(job)

        assert status == JobStatus.STOPPED

    async def test_long_running_job_shows_running(
        self,
        make_training_job: ...,
    ) -> None:
        """A sleeping job should be RUNNING or PENDING while alive."""
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job.submit_training()

        await asyncio.sleep(3)
        status = await job.get_training_status()

        assert status in (JobStatus.RUNNING, JobStatus.PENDING)

        await job.stop_training(timeout_seconds=15)

    async def test_failing_job_reaches_failed_status(
        self,
        make_training_job: ...,
    ) -> None:
        """A job that exits with nonzero should reach FAILED."""
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import sys; sys.exit(1)"')
        await job.submit_training()

        status = await poll_until_terminal(job)

        assert status == JobStatus.FAILED

    async def test_submit_raises_when_previous_job_active(
        self,
        make_training_job: ...,
    ) -> None:
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job.submit_training()

        with pytest.raises(RuntimeError, match="Cannot submit"):
            await job.submit_training()

        await job.stop_training(timeout_seconds=15)

    async def test_no_job_returns_stopped(
        self,
        make_training_job: ...,
    ) -> None:
        """get_training_status with no submitted job should return STOPPED."""
        job: RayTrainingJob = make_training_job()
        status = await job.get_training_status()

        assert status == JobStatus.STOPPED

    async def test_runtime_env_vars_visible_in_job(
        self,
        make_training_job: ...,
        job_client: ...,
    ) -> None:
        """The MILES_FT_TRAINING_RUN_ID env var should appear in the job logs."""
        job: RayTrainingJob = make_training_job(
            entrypoint='python -c "import os; print(os.environ[\'MILES_FT_TRAINING_RUN_ID\'])"',
        )
        run_id = await job.submit_training()

        await poll_until_terminal(job)

        logs = await asyncio.to_thread(job_client.get_job_logs, job.job_id)
        assert run_id in logs
