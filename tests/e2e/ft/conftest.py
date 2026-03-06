"""E2E test fixtures for FT system integration tests.

Each test is fully independent: clean environment → launch training via
launch_standard_run.main() → verify behavior → tear down.

Required environment variables:
  RAY_ADDRESS              — Ray cluster dashboard URL (e.g. http://head-node:8265)
  MILES_SCRIPT_EXTERNAL_RAY — Must be "1" (uses existing Ray cluster)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field

import pytest
import ray

from miles.utils.external_utils.command_utils import get_bool_env_var
from miles.utils.ft.fault_injectors.fault_injector import deploy_fault_injector
from miles.utils.ft.models import ControllerMode, ControllerStatus, RecoveryPhase
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)

_EXPECTED_CLUSTER_NODES = 3
_ACTOR_POLL_INTERVAL: float = 5.0


# ---------------------------------------------------------------------------
# Session-scoped: environment validation + Ray cluster connection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _assert_external_ray() -> None:
    assert get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
        "MILES_SCRIPT_EXTERNAL_RAY must be '1' for FT e2e tests"
    )


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
    """Shut down any leftover FtController and uncordon K8s nodes."""
    try:
        old_handle = ray.get_actor(ft_controller_actor_name(""))
        ray.get(old_handle.shutdown.remote(), timeout=60)
        logger.info("cleanup_shut_down_existing_controller")
    except ValueError:
        pass
    except Exception:
        logger.warning("cleanup_shutdown_existing_controller_failed", exc_info=True)

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
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return ray.get_actor(name)
        except ValueError as exc:
            last_error = exc
            await asyncio.sleep(_ACTOR_POLL_INTERVAL)

    raise TimeoutError(
        f"Named actor '{name}' did not appear within {timeout}s: {last_error}"
    )


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
    yield handle

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


@pytest.fixture
async def target_node(k8s_node_manager: K8sNodeManager) -> str:
    """Pick the first alive GPU node that is not already marked bad in K8s."""
    bad_nodes = set(await k8s_node_manager.get_bad_nodes())

    nodes = ray.nodes()
    candidates = [
        n for n in nodes
        if n.get("Alive")
        and n.get("Resources", {}).get("GPU", 0) > 0
        and n["NodeID"] not in bad_nodes
    ]
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
# Helper functions for E2E assertions
# ---------------------------------------------------------------------------


async def wait_for_recovery_complete(
    handle: ray.actor.ActorHandle,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll get_status() until mode returns to MONITORING."""
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        status = get_status(handle)
        poll_count += 1
        if status.mode == ControllerMode.MONITORING:
            return status
        if poll_count % 6 == 0:
            elapsed = timeout - (deadline - time.monotonic())
            logger.info(
                "wait_for_recovery_complete elapsed=%.0fs status=%s",
                elapsed, status,
            )
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Recovery did not complete within {timeout}s, "
        f"last status: {get_status(handle)}"
    )


async def wait_for_training_stable(
    handle: ray.actor.ActorHandle,
    n_iterations: int = 10,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> None:
    """Poll controller for N consecutive successful iterations."""
    baseline = get_iteration_count(handle)
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        current = get_iteration_count(handle)
        poll_count += 1
        if current - baseline >= n_iterations:
            return
        if poll_count % 6 == 0:
            elapsed = timeout - (deadline - time.monotonic())
            logger.info(
                "wait_for_training_stable elapsed=%.0fs progress=%d/%d",
                elapsed, current - baseline, n_iterations,
            )
        await asyncio.sleep(poll_interval)
    current = get_iteration_count(handle)
    raise TimeoutError(
        f"Training did not stabilize: need {n_iterations} iterations, "
        f"got {current - baseline} in {timeout}s"
    )


async def wait_for_recovery_phase(
    handle: ray.actor.ActorHandle,
    phase: RecoveryPhase,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll get_status() until recovery_phase matches."""
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        status = get_status(handle)
        poll_count += 1
        if status.recovery_phase == phase:
            return status
        if poll_count % 6 == 0:
            elapsed = timeout - (deadline - time.monotonic())
            logger.info(
                "wait_for_recovery_phase target='%s' elapsed=%.0fs status=%s",
                phase, elapsed, status,
            )
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Did not reach recovery phase '{phase}' within {timeout}s, "
        f"last status: {get_status(handle)}"
    )


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

    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode != target_mode:
            logger.info(
                "wait_for_mode_transition mode left '%s' → '%s'",
                target_mode, status.mode,
            )
            break
        await asyncio.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"Mode never left '{target_mode}' within {timeout}s, "
            f"last status: {get_status(handle)}"
        )

    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode == target_mode:
            return status
        await asyncio.sleep(poll_interval)

    raise TimeoutError(
        f"Mode did not return to '{target_mode}' within {timeout}s, "
        f"last status: {get_status(handle)}"
    )


def assert_phase_path_contains(
    status: ControllerStatus,
    required: list[RecoveryPhase],
) -> None:
    """Assert that phase_history contains the required phases in order (subsequence match)."""
    history = status.phase_history
    assert history is not None, "phase_history is None — was recovery never entered?"

    idx = 0
    for phase in history:
        if idx < len(required) and phase == required[idx]:
            idx += 1

    assert idx == len(required), (
        f"Expected phase path to contain {[p.value for p in required]} in order, "
        f"but got history {[p.value for p in history]}"
    )
