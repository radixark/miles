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
import signal
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
from tests.e2e.ft.utils import clear_all_bad_node_markers

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

_EXPECTED_CLUSTER_NODES = 4
_ACTOR_POLL_INTERVAL: float = 5.0


# ---------------------------------------------------------------------------
# Cluster topology definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterTopology:
    num_training_nodes: int
    rollout_num_cells: int


ROLLOUT_FOCUSED = ClusterTopology(num_training_nodes=1, rollout_num_cells=2)
TRAINING_FOCUSED = ClusterTopology(num_training_nodes=2, rollout_num_cells=1)
MULTI_CELL = ClusterTopology(num_training_nodes=1, rollout_num_cells=3)


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
        f"FT E2E tests require exactly {_EXPECTED_CLUSTER_NODES} cluster nodes, "
        f"got {len(alive_nodes)}"
    )

    gpu_node_list = [n for n in alive_nodes if n.get("Resources", {}).get("GPU", 0) > 0]
    assert len(gpu_node_list) == _EXPECTED_CLUSTER_NODES, (
        f"All {_EXPECTED_CLUSTER_NODES} cluster nodes must have GPUs, "
        f"but only {len(gpu_node_list)} of {len(alive_nodes)} have GPUs"
    )

    logger.info("ray_cluster_connected nodes=%d gpu_nodes=%d", len(alive_nodes), len(gpu_node_list))
    yield


# ---------------------------------------------------------------------------
# Environment cleanup helpers
# ---------------------------------------------------------------------------


async def _cleanup_sglang_processes() -> None:
    """Kill any leftover sglang processes on all cluster nodes."""
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node["NodeID"]
        try:
            injector = deploy_fault_injector(node_id=node_id)
            procs = ray.get(injector.find_sglang_processes.remote(), timeout=10)
            for proc in procs:
                try:
                    ray.get(injector.kill_process.remote(pid=proc["pid"], sig=signal.SIGKILL), timeout=5)
                    logger.info("cleanup_killed_sglang pid=%d node=%s", proc["pid"], node_id)
                except Exception:
                    logger.debug("cleanup_kill_sglang_failed pid=%d", proc["pid"], exc_info=True)
            ray.kill(injector)
        except Exception:
            logger.debug("cleanup_sglang_scan_failed node=%s", node_id, exc_info=True)


async def _cleanup_environment() -> None:
    """Shut down any leftover FtController, stop all Ray jobs, clean sglang, and uncordon K8s nodes."""
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

    await _cleanup_sglang_processes()

    node_mgr = K8sNodeManager(namespace=os.environ.get("K8S_NAMESPACE", "default"))
    try:
        await clear_all_bad_node_markers(node_mgr)
    except Exception:
        logger.warning("cleanup_clear_bad_node_markers_failed", exc_info=True)
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
    request: pytest.FixtureRequest,
) -> AsyncGenerator[ray.actor.ActorHandle, None]:
    """Launch independent training + FT Controller for a single test.

    Supports indirect parametrize via ClusterTopology:
      @pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)

    Default topology is TRAINING_FOCUSED when not parametrized.
    """
    topology: ClusterTopology = getattr(request, "param", TRAINING_FOCUSED)

    await _cleanup_environment()

    from tests.e2e.ft.launch_standard_run import main

    thread = threading.Thread(
        target=main,
        kwargs={
            "num_training_nodes": topology.num_training_nodes,
            "rollout_num_cells": topology.rollout_num_cells,
        },
        daemon=True,
    )
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
    node_mgr = K8sNodeManager(namespace=os.environ.get("K8S_NAMESPACE", "default"))
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


async def wait_for_sglang_pid(
    injector: ray.actor.ActorHandle,
    timeout: float = 60.0,
    poll_interval: float = 3.0,
) -> int:
    """Poll until a sglang process appears on the injector's node, then return its PID."""

    def _probe() -> int | None:
        procs = ray.get(injector.find_sglang_processes.remote())
        return procs[0]["pid"] if procs else None

    pid = await poll_until(
        probe=_probe,
        predicate=lambda p: p is not None,
        timeout=timeout,
        poll_interval=poll_interval,
        description="sglang_pid",
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


# ---------------------------------------------------------------------------
# Helper: subsystem state polling
# ---------------------------------------------------------------------------


async def wait_for_subsystem_state(
    handle: ray.actor.ActorHandle,
    subsystem_name: str,
    target_state: str,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll until a specific subsystem reaches the target state (class name)."""

    def _probe() -> ControllerStatus:
        return get_status(handle)

    return await poll_until(
        probe=_probe,
        predicate=lambda s: s.subsystem_states.get(subsystem_name) == target_state,
        timeout=timeout,
        poll_interval=poll_interval,
        description=f"subsystem_state({subsystem_name}={target_state})",
    )


async def wait_for_all_subsystems_detecting(
    handle: ray.actor.ActorHandle,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll until all subsystems are in DetectingAnomalySt."""

    def _all_detecting(status: ControllerStatus) -> bool:
        return bool(status.subsystem_states) and all(
            state == "DetectingAnomalySt" for state in status.subsystem_states.values()
        )

    return await poll_until(
        probe=lambda: get_status(handle),
        predicate=_all_detecting,
        timeout=timeout,
        poll_interval=poll_interval,
        description="all_subsystems_detecting",
    )


# ---------------------------------------------------------------------------
# Helper: rollout node discovery
# ---------------------------------------------------------------------------


def find_rollout_node(factory: FaultInjectorFactory) -> tuple[str, ray.actor.ActorHandle]:
    """Find one node running a sglang process. Returns (node_id, injector_handle)."""
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if not node.get("Resources", {}).get("GPU", 0) > 0:
            continue

        node_id = node["NodeID"]
        injector = factory.deploy_to(node_id=node_id)
        procs = ray.get(injector.find_sglang_processes.remote(), timeout=10)
        if procs:
            return node_id, injector

    raise AssertionError("No node running sglang found in the cluster")


def find_all_rollout_nodes(factory: FaultInjectorFactory) -> list[tuple[str, ray.actor.ActorHandle]]:
    """Find all nodes running sglang processes. Returns list of (node_id, injector_handle)."""
    results: list[tuple[str, ray.actor.ActorHandle]] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if not node.get("Resources", {}).get("GPU", 0) > 0:
            continue

        node_id = node["NodeID"]
        injector = factory.deploy_to(node_id=node_id)
        procs = ray.get(injector.find_sglang_processes.remote(), timeout=10)
        if procs:
            results.append((node_id, injector))

    return results


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


# ---------------------------------------------------------------------------
# E2E Fault Injector
# ---------------------------------------------------------------------------


class E2eFaultInjector:
    """FaultInjectionProtocol implementation for E2E tests.

    Wraps the remote FaultInjector actor to kill/stop real training processes.
    Also supports killing sglang processes on rollout nodes.
    """

    _DEFAULT_EXCEPTION_FLAG_PATH = "/tmp/miles_ft_inject_exception"

    def __init__(
        self,
        injector_handle: ray.actor.ActorHandle,
        target_node: str,
        exception_flag_path: str = _DEFAULT_EXCEPTION_FLAG_PATH,
        rollout_injectors: dict[str, ray.actor.ActorHandle] | None = None,
    ) -> None:
        self._injector = injector_handle
        self._target_node = target_node
        self._exception_flag_path = exception_flag_path
        self._rollout_injectors = rollout_injectors or {}

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

    async def crash_rollout_on_node(self, node_id: str) -> None:
        """SIGKILL the sglang process on a specific rollout node."""
        injector = self._rollout_injectors.get(node_id)
        assert injector is not None, (
            f"No fault injector deployed to rollout node {node_id}. "
            f"Available: {list(self._rollout_injectors.keys())}"
        )
        pid = await wait_for_sglang_pid(injector, timeout=60.0, poll_interval=3.0)
        ray.get(injector.kill_process.remote(pid=pid, sig=signal.SIGKILL))
        logger.info("crash_rollout_on_node node=%s pid=%d", node_id, pid)
