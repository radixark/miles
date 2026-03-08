"""Integration tests: stop_all_active_jobs against a real local Ray cluster."""

from __future__ import annotations

import asyncio

import pytest
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.adapters.impl.ray.training_job import RayTrainingJob, stop_all_active_jobs
from miles.utils.ft.adapters.types import JobStatus

pytestmark = [
    pytest.mark.local_ray,
]


class TestStopAllActiveJobs:
    async def test_stops_running_jobs(
        self,
        job_client: JobSubmissionClient,
        make_training_job: ...,
    ) -> None:
        # Step 1: submit two long-running jobs (via separate RayTrainingJob instances)
        job_a: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        job_b: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job_a.submit_training()
        await job_b.submit_training()

        await asyncio.sleep(2)

        # Step 2: stop all
        stopped = await stop_all_active_jobs(client=job_client, timeout_seconds=30)

        assert stopped >= 2

    async def test_returns_zero_when_no_active_jobs(
        self,
        job_client: JobSubmissionClient,
    ) -> None:
        stopped = await stop_all_active_jobs(client=job_client, timeout_seconds=15)
        assert stopped == 0

    async def test_idempotent_on_already_stopped(
        self,
        job_client: JobSubmissionClient,
        make_training_job: ...,
    ) -> None:
        # Step 1: submit and stop a job
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job.submit_training()
        await asyncio.sleep(2)
        await stop_all_active_jobs(client=job_client, timeout_seconds=30)

        # Step 2: call again — should be a no-op
        stopped = await stop_all_active_jobs(client=job_client, timeout_seconds=15)
        assert stopped == 0
