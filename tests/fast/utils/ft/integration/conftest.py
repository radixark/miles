from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import ray

from miles.utils.ft.controller.types import ControllerStatus

logger = logging.getLogger(__name__)

_TIMEOUT_SCALE = float(os.environ.get("FT_TEST_TIMEOUT_SCALE", "1.0"))
FAST_TIMEOUT = 30.0 * _TIMEOUT_SCALE
RECOVERY_TIMEOUT = 60.0 * _TIMEOUT_SCALE
LONG_RECOVERY_TIMEOUT = 120.0 * _TIMEOUT_SCALE
_WORKER_PORT_BLOCK_SIZE = 100
_WORKER_PORT_BASE = 20000


def _ray_start_env() -> dict[str, str]:
    env = os.environ.copy()
    env["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"
    return env


def _worker_port_range_args(node_index: int) -> list[str]:
    start_port = _WORKER_PORT_BASE + node_index * _WORKER_PORT_BLOCK_SIZE
    end_port = start_port + _WORKER_PORT_BLOCK_SIZE - 1
    return [
        f"--min-worker-port={start_port}",
        f"--max-worker-port={end_port}",
    ]


def _override_gcs_host(gcs_address: str, preferred_host: str | None) -> str:
    if preferred_host is None:
        return gcs_address

    host, sep, port = gcs_address.rpartition(":")
    if not sep or not host:
        return gcs_address
    return f"{preferred_host}:{port}"


def _normalize_local_ray_node_ip(node_ip: str, head_ip: str) -> str:
    if node_ip.startswith("127."):
        return node_ip
    return head_ip


def _connect_to_started_ray_cluster(
    start_stdout: str,
    preferred_host: str | None = None,
) -> tuple[Any, str]:
    match = re.search(r"--address='([^']+)'", start_stdout)
    if match:
        gcs_address = _override_gcs_host(match.group(1), preferred_host=preferred_host)
        return ray.init(address=gcs_address), gcs_address

    ctx = ray.init(address="auto")
    address_info = getattr(ctx, "address_info", {})
    gcs_address = address_info.get("gcs_address") or address_info.get("address")
    if not gcs_address:
        raise RuntimeError(
            "Could not resolve GCS address from ray start output or ray.init(address='auto'):\n"
            f"{start_stdout}"
        )
    gcs_address = _override_gcs_host(gcs_address, preferred_host=preferred_host)
    return ctx, gcs_address


def _init_local_ray() -> str:
    """Start a local Ray cluster with dashboard. Returns the dashboard URL.

    Uses ``ray start`` CLI instead of ``ray.init(address="local")`` because
    ``ray.init`` does not expose ``--dashboard-agent-listen-port``.  In
    ``--net=host`` containers the default ports (GCS 6379, agent 52365) are
    often already taken by another container, so we let the OS pick free ports.
    """
    if ray.is_initialized():
        ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    time.sleep(2)
    ray_tmp = Path("/tmp/ray")
    if ray_tmp.exists():
        shutil.rmtree(ray_tmp, ignore_errors=True)

    result = subprocess.run(
        [
            "ray",
            "start",
            "--head",
            "--port=0",
            "--num-cpus=32",
            "--num-gpus=0",
            "--include-dashboard=true",
            "--dashboard-host=127.0.0.1",
            "--dashboard-port=0",
            "--dashboard-agent-listen-port=0",
        ],
        check=True,
        capture_output=True,
        env=_ray_start_env(),
        text=True,
    )

    ctx, _ = _connect_to_started_ray_cluster(start_stdout=result.stdout)
    url = f"http://{ctx.dashboard_url}"
    _wait_for_dashboard_agent(url)
    return url


def _stop_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)


@pytest.fixture(scope="session")
def _ray_session() -> Generator[str, None, None]:
    """Session-scoped Ray cluster shared by all integration tests."""
    url = _init_local_ray()
    try:
        yield url
    finally:
        _stop_ray()


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
    raise RuntimeError(f"Ray dashboard agent not ready after {timeout}s: {last_error}")


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


# ------------------------------------------------------------------
# Multi-node Ray cluster (for MilesTestbed)
# ------------------------------------------------------------------


@dataclass
class RayNodeInfo:
    node_ip: str
    ray_node_id: str
    is_head: bool


_MULTI_NODE_COUNT = 5


def _dashboard_args(enabled: bool) -> list[str]:
    if enabled:
        return [
            "--include-dashboard=true",
            "--dashboard-host=127.0.0.1",
            "--dashboard-port=0",
            "--dashboard-agent-listen-port=0",
        ]

    return ["--include-dashboard=false"]


def _start_multi_node_ray(num_nodes: int = _MULTI_NODE_COUNT) -> list[RayNodeInfo]:
    """Start a multi-node Ray cluster using loopback aliases (127.0.0.x).

    Each "node" runs on a different loopback IP, which lets Ray treat them
    as separate nodes while staying on the same machine.
    """
    if ray.is_initialized():
        ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    time.sleep(2)

    temp_dir = Path(f"/tmp/ray_testbed_{uuid4().hex[:8]}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    head_ip = "127.0.0.1"
    result = subprocess.run(
        [
            "ray", "start", "--head",
            "--port=0",
            f"--node-ip-address={head_ip}",
            "--num-cpus=8",
            "--num-gpus=0",
            *_dashboard_args(enabled=False),
            *_worker_port_range_args(node_index=0),
            f"--temp-dir={temp_dir}",
        ],
        check=True,
        capture_output=True,
        env=_ray_start_env(),
        text=True,
    )

    ctx, gcs_address = _connect_to_started_ray_cluster(
        start_stdout=result.stdout,
        preferred_host=head_ip,
    )

    for i in range(1, num_nodes):
        node_ip = f"127.0.0.{i + 1}"
        subprocess.run(
            [
                "ray", "start",
                f"--address={gcs_address}",
                f"--node-ip-address={node_ip}",
                "--num-cpus=8",
                "--num-gpus=0",
                *_worker_port_range_args(node_index=i),
                f"--temp-dir={temp_dir}",
            ],
            check=True,
            capture_output=True,
            env=_ray_start_env(),
            text=True,
        )

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        nodes = ray.nodes()
        alive = [n for n in nodes if n.get("Alive", False)]
        if len(alive) >= num_nodes:
            break
        time.sleep(1.0)
    else:
        alive_count = len([n for n in ray.nodes() if n.get("Alive", False)])
        raise RuntimeError(
            f"Expected {num_nodes} alive Ray nodes, got {alive_count}"
        )

    node_infos: list[RayNodeInfo] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        node_ip = _normalize_local_ray_node_ip(
            node_ip=node["NodeManagerAddress"],
            head_ip=head_ip,
        )
        ray_node_id = node["NodeID"]
        is_head = node_ip == head_ip
        node_infos.append(RayNodeInfo(
            node_ip=node_ip,
            ray_node_id=ray_node_id,
            is_head=is_head,
        ))

    logger.info(
        "Multi-node Ray cluster started: %d nodes, temp_dir=%s",
        len(node_infos), temp_dir,
    )
    return node_infos


def _stop_multi_node_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)


@pytest.fixture(scope="session")
def local_ray_nodes() -> Generator[list[RayNodeInfo], None, None]:
    """Session-scoped multi-node Ray cluster for MilesTestbed tests.

    IMPORTANT: This fixture is incompatible with the single-node `local_ray`
    fixture. Tests using `local_ray_nodes` should NOT be run in the same
    pytest session as tests using `local_ray`. Use separate invocations or
    test selection (e.g. ``-k testbed`` vs ``-k "not testbed"``).
    """
    nodes = _start_multi_node_ray(num_nodes=_MULTI_NODE_COUNT)
    try:
        yield nodes
    finally:
        _stop_multi_node_ray()
