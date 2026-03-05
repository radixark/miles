from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import ray
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.platform.protocols import JobStatus

logger = logging.getLogger(__name__)


def resolve_to_ray_node_ids(identifiers: list[str]) -> list[str]:
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
        lookup[node.get("NodeName", "")] = ray_id
        lookup[node.get("NodeManagerAddress", "")] = ray_id

    resolved: list[str] = []
    for ident in identifiers:
        ray_id = lookup.get(ident)
        if ray_id is not None:
            resolved.append(ray_id)
        else:
            logger.warning("resolve_to_ray_node_ids: %s not found in Ray cluster, skipping", ident)
    return resolved

_RAY_STATUS_TO_JOB_STATUS: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "STOPPED": JobStatus.STOPPED,
    "SUCCEEDED": JobStatus.STOPPED,
    "FAILED": JobStatus.FAILED,
}

_TERMINAL_STATUSES = frozenset(("STOPPED", "FAILED", "SUCCEEDED"))

_DEFAULT_POLL_INTERVAL_SECONDS = 5
_DEFAULT_TIMEOUT_SECONDS = 300


def _parse_ray_status(raw_status: Any) -> str:
    return str(raw_status).rsplit(".", maxsplit=1)[-1]


class RayTrainingJob:
    """Manage training jobs via the Ray Job Submission API.

    Implements TrainingJobProtocol. All synchronous Ray client calls are
    wrapped with asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        client: JobSubmissionClient,
        entrypoint: str,
        runtime_env: dict[str, Any] | None = None,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
        ft_id: str = "",
        k8s_label_suffix: str = "",
    ) -> None:
        self._client = client
        self._entrypoint = entrypoint
        self._runtime_env = runtime_env or {}
        self._poll_interval = poll_interval_seconds
        self._ft_id = ft_id
        self._k8s_label_suffix = k8s_label_suffix
        self._job_id: str | None = None

    async def submit_training(
        self, excluded_node_ids: list[str] | None = None,
    ) -> str:
        run_id = uuid4().hex[:8]
        env_override = {
            **self._runtime_env.get("env_vars", {}),
            "FT_TRAINING_RUN_ID": run_id,
            "FT_ID": self._ft_id,
            "FT_K8S_LABEL_SUFFIX": self._k8s_label_suffix,
        }
        runtime_env = {**self._runtime_env, "env_vars": env_override}

        entrypoint = self._entrypoint
        if excluded_node_ids:
            ray_node_ids = resolve_to_ray_node_ids(excluded_node_ids)
            if ray_node_ids:
                entrypoint += f" --excluded-node-ids {','.join(ray_node_ids)}"

        start = time.monotonic()
        job_id = await asyncio.to_thread(
            self._client.submit_job,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
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

        start = time.monotonic()
        await asyncio.to_thread(self._client.stop_job, self._job_id)
        logger.info("stop_training_requested job_id=%s", self._job_id)

        deadline = start + timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {self._job_id} did not stop within {timeout_seconds}s")

            raw_status = await asyncio.to_thread(self._client.get_job_status, self._job_id)
            status_str = _parse_ray_status(raw_status)
            if status_str in _TERMINAL_STATUSES:
                elapsed = time.monotonic() - start
                logger.info(
                    "stop_training_completed job_id=%s final_status=%s elapsed_seconds=%.3f",
                    self._job_id,
                    status_str,
                    elapsed,
                )
                return

            await asyncio.sleep(self._poll_interval)

    async def get_training_status(self) -> JobStatus:
        if self._job_id is None:
            return JobStatus.STOPPED

        start = time.monotonic()
        raw_status = await asyncio.to_thread(self._client.get_job_status, self._job_id)
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
