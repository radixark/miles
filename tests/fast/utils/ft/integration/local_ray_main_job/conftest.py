"""Fixtures for RayMainJob integration tests against a real local Ray cluster."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob, stop_all_active_jobs
from miles.utils.ft.adapters.types import JobStatus

pytestmark = [
    pytest.mark.local_ray,
]

_POLL_INTERVAL = 0.5
_POLL_TIMEOUT = 30.0


@pytest.fixture
def job_client(local_ray_with_dashboard: str) -> JobSubmissionClient:
    return JobSubmissionClient(address=local_ray_with_dashboard)


@pytest.fixture
def make_main_job(
    job_client: JobSubmissionClient,
) -> Generator[..., None, None]:
    """Factory fixture that creates RayMainJob instances and cleans up after."""
    created: list[RayMainJob] = []

    def _factory(
        entrypoint: str = 'python -c "print(42)"',
        **kwargs: object,
    ) -> RayMainJob:
        job = RayMainJob(
            client=job_client,
            entrypoint=entrypoint,
            poll_interval_seconds=_POLL_INTERVAL,
            **kwargs,
        )
        created.append(job)
        return job

    yield _factory

    for job in created:
        if job.job_id is not None:
            try:
                asyncio.get_event_loop().run_until_complete(
                    job.stop_job(timeout_seconds=15)
                )
            except Exception:
                pass


async def poll_until_terminal(
    job: RayMainJob,
    *,
    timeout: float = _POLL_TIMEOUT,
    interval: float = _POLL_INTERVAL,
) -> JobStatus:
    """Poll get_job_status until a terminal state is reached."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await job.get_job_status()
        if status in (JobStatus.STOPPED, JobStatus.FAILED):
            return status
        await asyncio.sleep(interval)
    raise TimeoutError(f"Job did not reach terminal status within {timeout}s")
