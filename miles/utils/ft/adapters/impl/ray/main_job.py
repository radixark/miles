from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from ray.job_submission import JobSubmissionClient

from miles.utils.ft.adapters.types import STOP_TRAINING_TIMEOUT_SECONDS, JobStatus, MainJobProtocol
from miles.utils.ft.utils.polling import poll_until

logger = logging.getLogger(__name__)


_RAY_STATUS_TO_JOB_STATUS: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "STOPPED": JobStatus.STOPPED,
    "SUCCEEDED": JobStatus.STOPPED,
    "FAILED": JobStatus.FAILED,
}
_TERMINAL_STATUSES = frozenset(("STOPPED", "FAILED", "SUCCEEDED"))

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0

_SUBMIT_TIMEOUT_SECONDS = 60
_GET_STATUS_TIMEOUT_SECONDS = 30
_STOP_JOB_TIMEOUT_SECONDS = 30


async def stop_all_active_jobs(
    client: JobSubmissionClient,
    timeout_seconds: float = 60.0,
) -> int:
    """Stop all non-terminal Ray jobs. Returns count of jobs stopped."""
    all_jobs = await asyncio.to_thread(client.list_jobs)
    active = [j for j in all_jobs if _parse_ray_status(j.status) not in _TERMINAL_STATUSES]
    if not active:
        return 0

    stopped = 0
    for job in active:
        stop_id = getattr(job, "submission_id", None) or job.job_id
        if stop_id is None:
            continue
        try:
            await _stop_job(client, stop_id, timeout_seconds=timeout_seconds)
            stopped += 1
        except Exception:
            logger.warning("stop_all_active_jobs_failed job_id=%s", stop_id, exc_info=True)

    logger.info("stop_all_active_jobs stopped=%d attempted=%d", stopped, len(active))
    return stopped


class RayMainJob(MainJobProtocol):
    """Manage training jobs via the Ray Job Submission API.

    Implements MainJobProtocol. All synchronous Ray client calls are
    wrapped with asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        client: JobSubmissionClient,
        entrypoint: str,
        ft_id: str,
        k8s_label_prefix: str,
        runtime_env: dict[str, Any] | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        submit_timeout_seconds: float = _SUBMIT_TIMEOUT_SECONDS,
        get_status_timeout_seconds: float = _GET_STATUS_TIMEOUT_SECONDS,
        stop_job_timeout_seconds: float = _STOP_JOB_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._entrypoint = entrypoint
        self._runtime_env = runtime_env or {}
        self._poll_interval = poll_interval_seconds
        self._submit_timeout = submit_timeout_seconds
        self._get_status_timeout = get_status_timeout_seconds
        self._stop_job_timeout = stop_job_timeout_seconds
        self._ft_id = ft_id
        self._k8s_label_prefix = k8s_label_prefix
        self._job_id: str | None = None
        self._state_lock = asyncio.Lock()

    @property
    def job_id(self) -> str | None:
        return self._job_id

    async def start(self) -> str:
        async with self._state_lock:
            return await self._start_locked()

    async def stop(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None:
        async with self._state_lock:
            await self._stop_locked(timeout_seconds=timeout_seconds)

    async def get_status(self) -> JobStatus:
        async with self._state_lock:
            return await self._get_status_locked()

    async def _start_locked(self) -> str:
        if self._job_id is not None:
            raise RuntimeError(f"Cannot submit: previous job {self._job_id} still tracked. " "Call stop_job() first.")

        run_id = uuid4().hex[:8]
        env_override = {
            **self._runtime_env.get("env_vars", {}),
            "MILES_FT_TRAINING_RUN_ID": run_id,
            "MILES_FT_ID": self._ft_id,
            "MILES_FT_K8S_LABEL_PREFIX": self._k8s_label_prefix,
        }
        runtime_env = {**self._runtime_env, "env_vars": env_override}

        start = time.monotonic()
        job_id = await asyncio.wait_for(
            asyncio.to_thread(
                self._client.submit_job,
                entrypoint=self._entrypoint,
                runtime_env=runtime_env,
            ),
            timeout=self._submit_timeout,
        )
        elapsed = time.monotonic() - start

        self._job_id = job_id
        logger.info(
            "submit_job job_id=%s run_id=%s elapsed_seconds=%.3f",
            job_id,
            run_id,
            elapsed,
        )
        return run_id

    async def _stop_locked(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None:
        if self._job_id is None:
            logger.warning("stop_job called with no active job")
            return

        await _stop_job(
            client=self._client,
            job_id=self._job_id,
            timeout_seconds=timeout_seconds,
            poll_interval=self._poll_interval,
        )
        self._job_id = None

    async def _get_status_locked(self) -> JobStatus:
        if self._job_id is None:
            return JobStatus.STOPPED

        start = time.monotonic()
        raw_status = await asyncio.wait_for(
            asyncio.to_thread(self._client.get_job_status, self._job_id),
            timeout=self._get_status_timeout,
        )
        elapsed = time.monotonic() - start

        status_str = _parse_ray_status(raw_status)
        job_status = _RAY_STATUS_TO_JOB_STATUS.get(status_str)
        if job_status is None:
            logger.warning("unknown_ray_status raw_status=%s", status_str)
            job_status = JobStatus.FAILED

        logger.info(
            "get_job_status job_id=%s raw_status=%s job_status=%s elapsed_seconds=%.3f",
            self._job_id,
            status_str,
            job_status.value,
            elapsed,
        )
        return job_status


def _parse_ray_status(raw_status: Any) -> str:
    return str(raw_status).rsplit(".", maxsplit=1)[-1]


async def _stop_job(
    client: JobSubmissionClient,
    job_id: str,
    timeout_seconds: float = STOP_TRAINING_TIMEOUT_SECONDS,
    poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    """Stop a single Ray job and poll until it reaches terminal status."""
    start = time.monotonic()
    await asyncio.wait_for(
        asyncio.to_thread(client.stop_job, job_id),
        timeout=_STOP_JOB_TIMEOUT_SECONDS,
    )
    logger.info("stop_job_requested job_id=%s", job_id)

    async def _probe() -> str:
        raw_status = await asyncio.wait_for(
            asyncio.to_thread(client.get_job_status, job_id),
            timeout=_GET_STATUS_TIMEOUT_SECONDS,
        )
        return _parse_ray_status(raw_status)

    remaining = timeout_seconds - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError(f"stop_job({job_id}): no time left for polling after stop_job RPC")

    await poll_until(
        probe=_probe,
        predicate=lambda s: s in _TERMINAL_STATUSES,
        timeout=remaining,
        poll_interval=poll_interval,
        description=f"stop_job({job_id})",
    )

    elapsed = time.monotonic() - start
    logger.info("stop_job_completed job_id=%s elapsed_seconds=%.3f", job_id, elapsed)
