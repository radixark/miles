from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import ray
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.utils.polling import poll_until
from miles.utils.ft.protocols.platform import JobStatus, TrainingJobProtocol

logger = logging.getLogger(__name__)


def _resolve_to_ray_node_ids(identifiers: list[str]) -> list[str]:
    """Map node identifiers (K8s names, IPs, or Ray hex IDs) to Ray node IDs.

    Looks up each identifier against NodeName, NodeManagerAddress, and NodeID
    of alive Ray nodes. Identifiers already matching a NodeID pass through.
    Unresolvable identifiers are logged and skipped.
    """
    lookup: dict[str, str] = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        ray_id = node["NodeID"]
        lookup[ray_id] = ray_id
        if name := node.get("NodeName"):
            lookup[name] = ray_id
        if addr := node.get("NodeManagerAddress"):
            lookup[addr] = ray_id

    seen: set[str] = set()
    resolved: list[str] = []
    for ident in identifiers:
        ray_id = lookup.get(ident)
        if ray_id is not None and ray_id not in seen:
            seen.add(ray_id)
            resolved.append(ray_id)
        elif ray_id is None:
            logger.warning("_resolve_to_ray_node_ids: %s not found in Ray cluster, skipping", ident)
    return resolved


_RAY_STATUS_TO_JOB_STATUS: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "STOPPED": JobStatus.STOPPED,
    "SUCCEEDED": JobStatus.STOPPED,
    "FAILED": JobStatus.FAILED,
}

_TERMINAL_STATUSES = frozenset(("STOPPED", "FAILED", "SUCCEEDED"))

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_TIMEOUT_SECONDS = 300

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
        try:
            await _stop_job(client, job.job_id, timeout_seconds=timeout_seconds)
            stopped += 1
        except Exception:
            logger.warning("stop_all_active_jobs_failed job_id=%s", job.job_id, exc_info=True)

    logger.info("stop_all_active_jobs stopped=%d attempted=%d", stopped, len(active))
    return stopped


class RayTrainingJob(TrainingJobProtocol):
    """Manage training jobs via the Ray Job Submission API.

    Implements TrainingJobProtocol. All synchronous Ray client calls are
    wrapped with asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        client: JobSubmissionClient,
        entrypoint: str,
        runtime_env: dict[str, Any] | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        ft_id: str = "",
        k8s_label_prefix: str = "",
    ) -> None:
        self._client = client
        self._entrypoint = entrypoint
        self._runtime_env = runtime_env or {}
        self._poll_interval = poll_interval_seconds
        self._ft_id = ft_id
        self._k8s_label_prefix = k8s_label_prefix
        self._job_id: str | None = None

    @property
    def job_id(self) -> str | None:
        return self._job_id

    async def submit_training(
        self, excluded_node_ids: list[str] | None = None,
    ) -> str:
        if self._job_id is not None:
            raise RuntimeError(
                f"Cannot submit: previous job {self._job_id} still tracked. "
                "Call stop_training() first."
            )

        run_id = uuid4().hex[:8]
        env_override = {
            **self._runtime_env.get("env_vars", {}),
            "MILES_FT_TRAINING_RUN_ID": run_id,
            "MILES_FT_ID": self._ft_id,
            "MILES_FT_K8S_LABEL_PREFIX": self._k8s_label_prefix,
        }
        runtime_env = {**self._runtime_env, "env_vars": env_override}

        entrypoint = self._entrypoint
        if excluded_node_ids:
            ray_node_ids = await asyncio.to_thread(_resolve_to_ray_node_ids, excluded_node_ids)
            if ray_node_ids:
                entrypoint += f" --excluded-node-ids {','.join(ray_node_ids)}"

        start = time.monotonic()
        job_id = await asyncio.wait_for(
            asyncio.to_thread(
                self._client.submit_job,
                entrypoint=entrypoint,
                runtime_env=runtime_env,
            ),
            timeout=_SUBMIT_TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - start

        self._job_id = job_id
        logger.info(
            "submit_training job_id=%s run_id=%s elapsed_seconds=%.3f",
            job_id,
            run_id,
            elapsed,
        )
        return run_id

    async def stop_training(self, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
        if self._job_id is None:
            logger.warning("stop_training called with no active job")
            return

        await _stop_job(
            client=self._client,
            job_id=self._job_id,
            timeout_seconds=timeout_seconds,
            poll_interval=self._poll_interval,
        )
        self._job_id = None

    async def get_training_status(self) -> JobStatus:
        if self._job_id is None:
            return JobStatus.STOPPED

        start = time.monotonic()
        raw_status = await asyncio.wait_for(
            asyncio.to_thread(self._client.get_job_status, self._job_id),
            timeout=_GET_STATUS_TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - start

        status_str = _parse_ray_status(raw_status)
        job_status = _RAY_STATUS_TO_JOB_STATUS.get(status_str)
        if job_status is None:
            logger.warning("unknown_ray_status raw_status=%s", status_str)
            job_status = JobStatus.FAILED

        logger.info(
            "get_training_status job_id=%s raw_status=%s job_status=%s elapsed_seconds=%.3f",
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
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
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
        raise TimeoutError(
            f"stop_job({job_id}): no time left for polling after stop_job RPC"
        )

    await poll_until(
        probe=_probe,
        predicate=lambda s: s in _TERMINAL_STATUSES,
        timeout=remaining,
        poll_interval=poll_interval,
        description=f"stop_job({job_id})",
    )

    elapsed = time.monotonic() - start
    logger.info("stop_job_completed job_id=%s elapsed_seconds=%.3f", job_id, elapsed)
