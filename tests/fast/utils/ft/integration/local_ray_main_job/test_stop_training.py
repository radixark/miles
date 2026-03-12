"""Integration tests: RayMainJob stop_job against a real local Ray cluster."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.integration.local_ray_main_job.conftest import poll_until_terminal

from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob
from miles.utils.ft.adapters.types import JobStatus

pytestmark = [
    pytest.mark.local_ray,
]


class TestStopJob:
    async def test_stop_running_job_reaches_terminal(
        self,
        make_main_job: ...,
    ) -> None:
        job: RayMainJob = make_main_job(entrypoint='python -c "import time; time.sleep(300)"')
        await job.start()

        await job.stop(timeout_seconds=30)

        status = await job.get_status()
        assert status == JobStatus.STOPPED

    async def test_submit_after_stop_creates_new_job(
        self,
        make_main_job: ...,
    ) -> None:
        job: RayMainJob = make_main_job(entrypoint='python -c "import time; time.sleep(300)"')

        # Step 1: submit and stop
        run_id_1 = await job.start()
        await job.stop(timeout_seconds=30)

        # Step 2: resubmit
        run_id_2 = await job.start()
        assert run_id_1 != run_id_2

        await job.stop(timeout_seconds=30)

    async def test_stop_with_no_active_job_is_noop(
        self,
        make_main_job: ...,
    ) -> None:
        job: RayMainJob = make_main_job()

        await job.stop(timeout_seconds=15)

    async def test_stop_already_succeeded_job_completes_immediately(
        self,
        make_main_job: ...,
    ) -> None:
        """Stopping a job that already SUCCEEDED should complete without error."""
        job: RayMainJob = make_main_job(entrypoint='python -c "print(42)"')
        await job.start()

        await poll_until_terminal(job)

        await job.stop(timeout_seconds=15)
        assert job.job_id is None
