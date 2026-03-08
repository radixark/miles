"""Integration tests: RayTrainingJob stop_training against a real local Ray cluster."""

from __future__ import annotations

import pytest

from miles.utils.ft.adapters.impl.ray.training_job import RayTrainingJob
from miles.utils.ft.adapters.types import JobStatus
from tests.fast.utils.ft.integration.local_ray_training_job.conftest import poll_until_terminal

pytestmark = [
    pytest.mark.local_ray,
]


class TestStopTraining:
    async def test_stop_running_job_reaches_terminal(
        self,
        make_training_job: ...,
    ) -> None:
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job.submit_training()

        await job.stop_training(timeout_seconds=30)

        status = await job.get_training_status()
        assert status == JobStatus.STOPPED

    async def test_submit_after_stop_creates_new_job(
        self,
        make_training_job: ...,
    ) -> None:
        job: RayTrainingJob = make_training_job(entrypoint='python -c "import time; time.sleep(300)"')

        # Step 1: submit and stop
        run_id_1 = await job.submit_training()
        await job.stop_training(timeout_seconds=30)

        # Step 2: resubmit
        run_id_2 = await job.submit_training()
        assert run_id_1 != run_id_2

        await job.stop_training(timeout_seconds=30)

    async def test_stop_with_no_active_job_is_noop(
        self,
        make_training_job: ...,
    ) -> None:
        job: RayTrainingJob = make_training_job()

        await job.stop_training(timeout_seconds=15)

    async def test_stop_already_succeeded_job_completes_immediately(
        self,
        make_training_job: ...,
    ) -> None:
        """Stopping a job that already SUCCEEDED should complete without error."""
        job: RayTrainingJob = make_training_job(entrypoint='python -c "print(42)"')
        await job.submit_training()

        await poll_until_terminal(job)

        await job.stop_training(timeout_seconds=15)
        assert job.job_id is None
