from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import ray

from miles.utils.ft.controller.types import ControllerStatus

logger = logging.getLogger(__name__)

_TIMEOUT_SCALE = float(os.environ.get("FT_TEST_TIMEOUT_SCALE", "1.0"))
FAST_TIMEOUT = 30.0 * _TIMEOUT_SCALE
RECOVERY_TIMEOUT = 60.0 * _TIMEOUT_SCALE
LONG_RECOVERY_TIMEOUT = 120.0 * _TIMEOUT_SCALE


def _init_local_ray() -> str:
    """Start a local Ray cluster with dashboard. Returns the dashboard URL."""
    if ray.is_initialized():
        ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    time.sleep(2)
    ray_tmp = Path("/tmp/ray")
    if ray_tmp.exists():
        shutil.rmtree(ray_tmp, ignore_errors=True)

    # In --net=host containers the default dashboard-agent port (52365) is
    # often already taken by another container.  Let the OS pick a free port.
    os.environ.setdefault("RAY_DASHBOARD_AGENT_LISTEN_PORT", "0")

    ctx = ray.init(
        address="local",
        num_cpus=32,
        num_gpus=0,
        include_dashboard=True,
        dashboard_host="127.0.0.1",
        dashboard_port=0,
    )
    url = f"http://{ctx.dashboard_url}"
    _wait_for_dashboard_agent(url)
    return url


@pytest.fixture(scope="session")
def _ray_session() -> Generator[str, None, None]:
    """Session-scoped Ray cluster shared by all integration tests."""
    url = _init_local_ray()
    yield url
    ray.shutdown()


@pytest.fixture(scope="session")
def local_ray(_ray_session: str) -> Generator[None, None, None]:
    yield


@pytest.fixture(scope="session")
def local_ray_with_dashboard(_ray_session: str) -> Generator[str, None, None]:
    yield _ray_session


_DASHBOARD_AGENT_TIMEOUT = 90.0


def _wait_for_dashboard_agent(dashboard_url: str, timeout: float = _DASHBOARD_AGENT_TIMEOUT) -> None:
    """Wait until the Ray dashboard job agent can accept job submissions.

    Submits a trivial probe job to verify end-to-end readiness.
    """
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(address=dashboard_url)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            job_id = client.submit_job(entrypoint='python -c "1"')
            client.stop_job(job_id)
            logger.info("Dashboard agent ready after probe job %s", job_id)
            return
        except Exception as exc:
            last_error = exc
            logger.debug("Dashboard agent not ready yet", exc_info=True)
        time.sleep(2.0)
    raise RuntimeError(
        f"Ray dashboard agent not ready after {timeout}s: {last_error}"
    )


def _kill_named_actor(name: str) -> None:
    try:
        handle = ray.get_actor(name)
        ray.kill(handle, no_restart=True)
    except ValueError:
        pass
    except Exception:
        logger.warning("Failed to kill actor %s", name, exc_info=True)


def get_status(handle: ray.actor.ActorHandle, timeout: float = 5) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


def poll_for_run_id(
    handle: ray.actor.ActorHandle,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> str:
    """Poll get_status until active_run_id is set, return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.active_run_id is not None:
            return status.active_run_id
        time.sleep(interval)
    raise TimeoutError("active_run_id not set within timeout")
