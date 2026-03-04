from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import structlog
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.platform.protocols import JobStatus, TrainingJobProtocol

log = structlog.get_logger(__name__)

_RAY_STATUS_TO_JOB_STATUS: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "STOPPED": JobStatus.STOPPED,
    "SUCCEEDED": JobStatus.STOPPED,
    "FAILED": JobStatus.FAILED,
}

_DEFAULT_POLL_INTERVAL_SECONDS = 5
_DEFAULT_TIMEOUT_SECONDS = 300


class RayTrainingJob:
    """Manage training jobs via the Ray Job Submission API.

    Implements TrainingJobProtocol. All synchronous Ray client calls are
    wrapped with asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        client: JobSubmissionClient,
        entrypoint: str,
        runtime_env: dict | None = None,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._entrypoint = entrypoint
        self._runtime_env = runtime_env or {}
        self._poll_interval = poll_interval_seconds
        self._job_id: str | None = None

    async def submit_training(self) -> str:
        run_id = uuid4().hex[:8]
        env_override = {
            **self._runtime_env.get("env_vars", {}),
            "FT_TRAINING_RUN_ID": run_id,
        }
        runtime_env = {**self._runtime_env, "env_vars": env_override}

        start = time.monotonic()
        job_id = await asyncio.to_thread(
            self._client.submit_job,
            entrypoint=self._entrypoint,
            runtime_env=runtime_env,
        )
        elapsed = time.monotonic() - start

        self._job_id = job_id
        log.info(
            "submit_training",
            job_id=job_id,
            run_id=run_id,
            elapsed_seconds=round(elapsed, 3),
        )
        return run_id

    async def stop_training(self, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
        if self._job_id is None:
            log.warning("stop_training called with no active job")
            return

        start = time.monotonic()
        await asyncio.to_thread(self._client.stop_job, self._job_id)
        log.info("stop_training_requested", job_id=self._job_id)

        deadline = start + timeout_seconds
        while True:
            raw_status = await asyncio.to_thread(
                self._client.get_job_status, self._job_id
            )
            status_str = str(raw_status).rsplit(".", maxsplit=1)[-1]
            if status_str in ("STOPPED", "FAILED", "SUCCEEDED"):
                elapsed = time.monotonic() - start
                log.info(
                    "stop_training_completed",
                    job_id=self._job_id,
                    final_status=status_str,
                    elapsed_seconds=round(elapsed, 3),
                )
                return

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {self._job_id} did not stop within {timeout_seconds}s "
                    f"(last status: {status_str})"
                )

            await asyncio.sleep(self._poll_interval)

    async def get_training_status(self) -> JobStatus:
        if self._job_id is None:
            return JobStatus.STOPPED

        start = time.monotonic()
        raw_status = await asyncio.to_thread(
            self._client.get_job_status, self._job_id
        )
        elapsed = time.monotonic() - start

        status_str = str(raw_status).rsplit(".", maxsplit=1)[-1]
        job_status = _RAY_STATUS_TO_JOB_STATUS.get(status_str, JobStatus.FAILED)

        log.info(
            "get_training_status",
            job_id=self._job_id,
            raw_status=status_str,
            job_status=job_status.value,
            elapsed_seconds=round(elapsed, 3),
        )
        return job_status


def _check_protocol_conformance() -> None:
    _: type[TrainingJobProtocol] = RayTrainingJob  # type: ignore[assignment]
