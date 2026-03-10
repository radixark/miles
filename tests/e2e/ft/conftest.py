"""E2E test fixtures for FT system integration tests.

Each test is fully independent: clean environment → launch training via
launch_standard_run.main() → verify behavior → tear down.

Required environment variables:
  RAY_ADDRESS              — Ray cluster dashboard URL (e.g. http://head-node:8265)
  MILES_SCRIPT_EXTERNAL_RAY — Must be "1" (uses existing Ray cluster)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
import ray
from ray.job_submission import JobSubmissionClient
from tests.fast.utils.ft.integration.local_ray_semi_e2e import scenarios as _scenarios

from miles.utils.external_utils.command_utils import get_bool_env_var
from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
from miles.utils.ft.adapters.impl.ray.main_job import stop_all_active_jobs
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.controller.types import ControllerMode, ControllerStatus
from miles.utils.ft.fault_injectors.fault_injector import deploy_fault_injector
from miles.utils.ft.utils.polling import poll_until

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]

_EXPECTED_CLUSTER_NODES = 3
_ACTOR_POLL_INTERVAL: float = 5.0


# ---------------------------------------------------------------------------
# Session-scoped: environment validation + Ray cluster connection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _k8s_label_prefix() -> Generator[str, None, None]:
    """Generate a random prefix for K8s node labels to isolate concurrent test sessions."""
    prefix = uuid4().hex[:8]
    old = os.environ.get("MILES_FT_K8S_LABEL_PREFIX")
    os.environ["MILES_FT_K8S_LABEL_PREFIX"] = prefix
    logger.info("k8s_label_prefix=%s", prefix)
    yield prefix
    if old is None:
        os.environ.pop("MILES_FT_K8S_LABEL_PREFIX", None)
    else:
        os.environ["MILES_FT_K8S_LABEL_PREFIX"] = old


@pytest.fixture(scope="session", autouse=True)
def _assert_external_ray() -> None:
    assert get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), "MILES_SCRIPT_EXTERNAL_RAY must be '1' for FT e2e tests"


@pytest.fixture(scope="session")
def ray_address() -> str:
    addr = os.environ.get("RAY_ADDRESS", "").strip()
    if not addr:
        pytest.skip("RAY_ADDRESS not set — skipping E2E tests")
    return addr


@pytest.fixture(scope="session")
def ray_cluster(ray_address: str) -> Generator[None, None, None]:
    """Connect to existing Ray cluster and validate node count."""
    if not ray.is_initialized():
        ray.init(address=ray_address)

    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive")]
    assert len(alive_nodes) == _EXPECTED_CLUSTER_NODES, (
        f"FT E2E tests require exactly {_EXPECTED_CLUSTER_NODES} cluster nodes "
        f"(2 training + 1 spare for eviction), got {len(alive_nodes)}"
    )

    gpu_nodes = [n for n in alive_nodes if n.get("Resources", {}).get("GPU", 0) > 0]
    assert len(gpu_nodes) == _EXPECTED_CLUSTER_NODES, (
        f"All {_EXPECTED_CLUSTER_NODES} cluster nodes must have GPUs, "
        f"but only {len(gpu_nodes)} of {len(alive_nodes)} have GPUs"
    )

    logger.info("ray_cluster_connected nodes=%d gpu_nodes=%d", len(alive_nodes), len(gpu_nodes))
    yield


# ---------------------------------------------------------------------------
# Environment cleanup helpers
# ---------------------------------------------------------------------------


async def _cleanup_environment() -> None:
    """Shut down any leftover FtController, stop all Ray jobs, and uncordon K8s nodes."""
    try:
        old_handle = ray.get_actor(ft_controller_actor_name(""))
        ray.get(old_handle.shutdown.remote(), timeout=60)
        logger.info("cleanup_shut_down_existing_controller")
    except ValueError:
        logger.debug("cleanup_no_existing_controller")
    except Exception:
        logger.warning("cleanup_shutdown_existing_controller_failed", exc_info=True)

    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if ray_address:
        try:
            client = JobSubmissionClient(address=ray_address)
            await stop_all_active_jobs(client)
        except Exception:
            logger.warning("cleanup_stop_all_jobs_failed", exc_info=True)

    node_mgr = K8sNodeManager()
    try:
        bad_nodes = await node_mgr.get_bad_nodes()
        for node_id in bad_nodes:
            try:
                await node_mgr.unmark_node_bad(node_id=node_id)
            except Exception:
                logger.warning("cleanup_uncordon_failed node_id=%s", node_id, exc_info=True)
    except Exception:
        logger.warning("cleanup_get_bad_nodes_failed", exc_info=True)
    finally:
        await node_mgr.aclose()


async def _wait_for_named_actor(
    name: str,
    timeout: float,
) -> ray.actor.ActorHandle:
    """Poll until a named Ray actor becomes available."""

    def _probe() -> ray.actor.ActorHandle | None:
        try:
            return ray.get_actor(name)
        except ValueError:
            return None

    handle = await poll_until(
        probe=_probe,
        predicate=lambda h: h is not None,
        timeout=timeout,
        poll_interval=_ACTOR_POLL_INTERVAL,
        description=f"named_actor({name})",
    )
    assert handle is not None
    return handle


# ---------------------------------------------------------------------------
# Function-scoped: independent training launch per test
# ---------------------------------------------------------------------------


@pytest.fixture
async def ft_controller_handle(
    ray_cluster: None,
) -> AsyncGenerator[ray.actor.ActorHandle, None]:
    """Launch independent training + FT Controller for a single test.

    1. Clean up leftover state (controller, K8s node cordons)
    2. Launch launch_standard_run.main() in background thread
    3. Wait for FtController actor to appear
    4. Yield controller handle
    5. Tear down controller and clean up
    """
    await _cleanup_environment()

    from tests.e2e.ft.launch_standard_run import main

    thread = threading.Thread(target=main, daemon=True)
    thread.start()

    handle = await _wait_for_named_actor(
        name=ft_controller_actor_name(""),
        timeout=300.0,
    )

    try:
        yield handle
    finally:
        try:
            ray.get(handle.shutdown.remote(), timeout=60)
        except Exception:
            logger.warning("ft_controller_teardown_failed", exc_info=True)

        await _cleanup_environment()


# ---------------------------------------------------------------------------
# Function-scoped: K8sNodeManager
# ---------------------------------------------------------------------------


@pytest.fixture
async def k8s_node_manager(ray_cluster: None) -> AsyncGenerator[K8sNodeManager, None]:
    """Shared K8sNodeManager for test use (e.g. checking bad nodes)."""
    node_mgr = K8sNodeManager()
    yield node_mgr
    await node_mgr.aclose()


# ---------------------------------------------------------------------------
# Function-scoped: Fault injector factory
# ---------------------------------------------------------------------------


@dataclass
class FaultInjectorFactory:
    _injectors: list[ray.actor.ActorHandle] = field(default_factory=list)

    def deploy_to(self, node_id: str) -> ray.actor.ActorHandle:
        handle = deploy_fault_injector(node_id=node_id)
        self._injectors.append(handle)
        return handle

    def cleanup_all(self) -> None:
        for handle in self._injectors:
            try:
                ray.get(handle.cleanup_all.remote(), timeout=30)
            except Exception:
                logger.warning("fault_injector_cleanup_failed", exc_info=True)
        self._injectors.clear()


@pytest.fixture
def fault_injector(ray_cluster: None) -> Generator[FaultInjectorFactory, None, None]:
    factory = FaultInjectorFactory()
    yield factory
    factory.cleanup_all()


# ---------------------------------------------------------------------------
# Function-scoped: Target node selection
# ---------------------------------------------------------------------------


def gpu_nodes(*, exclude: set[str] | None = None) -> list[dict]:
    """Return alive Ray nodes that have GPUs, optionally excluding node IDs."""
    return [
        n
        for n in ray.nodes()
        if n.get("Alive")
        and n.get("Resources", {}).get("GPU", 0) > 0
        and (exclude is None or n["NodeID"] not in exclude)
    ]


@pytest.fixture
async def target_node(k8s_node_manager: K8sNodeManager) -> str:
    """Pick the first alive GPU node that is not already marked bad in K8s."""
    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    candidates = gpu_nodes(exclude=bad_nodes)
    if not candidates:
        pytest.skip("No available healthy GPU nodes")
    return candidates[0]["NodeID"]


# ---------------------------------------------------------------------------
# Helper: query controller status via actor handle
# ---------------------------------------------------------------------------


def get_status(handle: ray.actor.ActorHandle) -> ControllerStatus:
    """Synchronous wrapper: query controller status via the Ray actor."""
    return ray.get(handle.get_status.remote())


def get_iteration_count(handle: ray.actor.ActorHandle) -> int:
    """Query current iteration from the controller."""
    status = get_status(handle)
    return status.latest_iteration or 0


# ---------------------------------------------------------------------------
# Helper: fault injection utilities
# ---------------------------------------------------------------------------


def find_training_pid(injector: ray.actor.ActorHandle, node_id: str = "") -> int:
    """Find first training process PID on the injector's node. Fails if none found."""
    procs = ray.get(injector.find_training_processes.remote())
    assert procs, f"No training processes found{f' on node {node_id}' if node_id else ''}"
    return procs[0]["pid"]


async def wait_for_training_pid(
    injector: ray.actor.ActorHandle,
    timeout: float = 30.0,
    poll_interval: float = 3.0,
) -> int:
    """Poll until a training process appears on the injector's node, then return its PID."""

    def _probe() -> int | None:
        procs = ray.get(injector.find_training_processes.remote())
        return procs[0]["pid"] if procs else None

    pid = await poll_until(
        probe=_probe,
        predicate=lambda p: p is not None,
        timeout=timeout,
        poll_interval=poll_interval,
        description="training_pid",
    )
    assert pid is not None
    return pid


# ---------------------------------------------------------------------------
# Helper functions for E2E assertions
# ---------------------------------------------------------------------------


wait_for_recovery_complete = _scenarios.wait_for_recovery_complete


async def wait_for_training_stable(
    handle: ray.actor.ActorHandle,
    n_iterations: int = 5,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> None:
    """Poll controller for N consecutive successful iterations."""
    baseline = get_iteration_count(handle)
    await poll_until(
        probe=lambda: get_iteration_count(handle),
        predicate=lambda current: current - baseline >= n_iterations,
        timeout=timeout,
        poll_interval=poll_interval,
        description=f"training_stable(need={n_iterations})",
    )


wait_for_recovery_phase = _scenarios.wait_for_recovery_phase


async def wait_for_mode_transition(
    handle: ray.actor.ActorHandle,
    target_mode: ControllerMode,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Wait for mode to leave *target_mode*, then return to it.

    Avoids the race where polling for a mode that is already active
    returns immediately before the fault has been detected.
    """
    deadline = time.monotonic() + timeout

    await poll_until(
        probe=lambda: get_status(handle),
        predicate=lambda s: s.mode != target_mode,
        timeout=deadline - time.monotonic(),
        poll_interval=poll_interval,
        description=f"mode_leave({target_mode})",
    )

    return await poll_until(
        probe=lambda: get_status(handle),
        predicate=lambda s: s.mode == target_mode,
        timeout=deadline - time.monotonic(),
        poll_interval=poll_interval,
        description=f"mode_return({target_mode})",
    )


assert_phase_path_contains = _scenarios.assert_phase_path_contains


async def list_worker_pods_on_node(node_id: str, namespace: str = "default") -> list[str]:
    """List ray worker pod names scheduled on a specific K8s node."""
    from kubernetes_asyncio import config as k8s_config
    from kubernetes_asyncio.client import ApiClient, CoreV1Api

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        await k8s_config.load_kube_config()

    async with ApiClient() as api_client:
        core_v1 = CoreV1Api(api_client)
        pod_list = await core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector="ray.io/node-type=worker",
            field_selector=f"spec.nodeName={node_id}",
        )
        return [pod.metadata.name for pod in pod_list.items]


class E2eFaultInjector:
    """FaultInjectionProtocol implementation for E2E tests.

    Wraps the remote FaultInjector actor to kill/stop real training processes.
    """

    _DEFAULT_EXCEPTION_FLAG_PATH = "/tmp/miles_ft_inject_exception"

    def __init__(
        self,
        injector_handle: ray.actor.ActorHandle,
        target_node: str,
        exception_flag_path: str = _DEFAULT_EXCEPTION_FLAG_PATH,
    ) -> None:
        self._injector = injector_handle
        self._target_node = target_node
        self._exception_flag_path = exception_flag_path

    async def crash_training(self) -> None:
        pid = await wait_for_training_pid(
            self._injector,
            timeout=60.0,
            poll_interval=3.0,
        )
        ray.get(self._injector.kill_process.remote(pid=pid, sig=9))

    async def recover_training(self) -> None:
        pass

    async def inject_hang(self) -> None:
        pid = find_training_pid(self._injector, node_id=self._target_node)
        ray.get(self._injector.stop_process.remote(pid=pid))

    async def inject_python_exception(self) -> None:
        ray.get(self._injector.write_exception_flag.remote(path=self._exception_flag_path))
