"""E2E test fixtures for FT system integration tests.

Required environment variables:
  RAY_ADDRESS         — Ray cluster dashboard URL (e.g. http://head-node:8265)
  FT_E2E_TRAINING_ENTRYPOINT — Training job entrypoint command

Required cluster: >= 4 GPU nodes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field

import pytest
import ray
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.fault_injectors.fault_injector import deploy_fault_injector
from miles.utils.ft.models import ControllerMode, ControllerStatus, RecoveryPhase
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from miles.utils.ft.platform.ray_training_job import RayTrainingJob

logger = logging.getLogger(__name__)

_MIN_CLUSTER_NODES = 4


# ---------------------------------------------------------------------------
# Session-scoped: Ray cluster
# ---------------------------------------------------------------------------


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
    if len(alive_nodes) < _MIN_CLUSTER_NODES:
        pytest.skip(
            f"Need >= {_MIN_CLUSTER_NODES} alive nodes, got {len(alive_nodes)}"
        )

    logger.info("ray_cluster_connected nodes=%d", len(alive_nodes))
    yield
    # Don't disconnect — session-level


# ---------------------------------------------------------------------------
# Function-scoped: FT system (Controller + metric store)
# ---------------------------------------------------------------------------


@dataclass
class FtSystem:
    controller: FtController
    metric_store: MiniPrometheus
    mini_wandb: MiniWandb
    training_job: RayTrainingJob
    node_manager: K8sNodeManager
    _controller_task: asyncio.Task[None] | None = field(default=None, repr=False)

    async def start(self) -> None:
        self._controller_task = asyncio.create_task(self.controller.run())

    async def shutdown(self) -> None:
        await self.controller.shutdown()
        if self._controller_task is not None:
            self._controller_task.cancel()
            try:
                await self._controller_task
            except asyncio.CancelledError:
                pass


@pytest.fixture
async def ft_system(
    ray_cluster: None, ray_address: str,
) -> AsyncGenerator[FtSystem, None]:
    """Start a fresh FT Controller with mini metric store for each test."""
    entrypoint = os.environ.get("FT_E2E_TRAINING_ENTRYPOINT", "").strip()
    if not entrypoint:
        pytest.skip("FT_E2E_TRAINING_ENTRYPOINT not set")

    controller_exporter = ControllerExporter(port=0)
    mini_prom = MiniPrometheus(config=MiniPrometheusConfig())
    mini_prom.add_scrape_target(
        target_id="controller",
        address=controller_exporter.address,
    )
    mini_wandb = MiniWandb()
    rank_registry = RankRegistry(
        mini_wandb=mini_wandb,
        scrape_target_manager=mini_prom,
    )
    node_manager = K8sNodeManager()
    training_job = RayTrainingJob(
        client=JobSubmissionClient(address=ray_address),
        entrypoint=entrypoint,
    )

    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=mini_prom,
        rank_registry=rank_registry,
        detectors=build_detector_chain(),
        tick_interval=5.0,
        controller_exporter=controller_exporter,
    )

    system = FtSystem(
        controller=controller,
        metric_store=mini_prom,
        mini_wandb=mini_wandb,
        training_job=training_job,
        node_manager=node_manager,
    )

    await system.start()
    yield system
    await system.shutdown()
    try:
        await training_job.stop_training()
    except Exception:
        logger.warning("ft_system_teardown_stop_training_failed", exc_info=True)
    await node_manager.aclose()


# ---------------------------------------------------------------------------
# Function-scoped: Shared K8sNodeManager for cleanup & node selection
# ---------------------------------------------------------------------------


@pytest.fixture
async def _cleanup_node_manager(ray_cluster: None) -> AsyncGenerator[K8sNodeManager, None]:
    """Shared K8sNodeManager for test-infrastructure fixtures (not the SUT)."""
    node_mgr = K8sNodeManager()
    yield node_mgr
    await node_mgr.aclose()


@pytest.fixture(autouse=True)
async def _restore_cluster_state(
    _cleanup_node_manager: K8sNodeManager,
) -> AsyncGenerator[None, None]:
    """Uncordon any nodes marked bad during the test, even on failure."""
    yield
    try:
        bad_nodes = await _cleanup_node_manager.get_bad_nodes()
        for node_id in bad_nodes:
            try:
                await _cleanup_node_manager.unmark_node_bad(node_id=node_id)
            except Exception:
                logger.warning(
                    "restore_cluster_unmark_failed node_id=%s", node_id,
                    exc_info=True,
                )
    except Exception:
        logger.warning("restore_cluster_get_bad_nodes_failed", exc_info=True)


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
async def target_node(_cleanup_node_manager: K8sNodeManager) -> str:
    """Pick the first alive GPU node that is not already marked bad in K8s."""
    bad_nodes = set(await _cleanup_node_manager.get_bad_nodes())

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
# Helper functions for E2E assertions
# ---------------------------------------------------------------------------


async def wait_for_recovery_complete(
    controller: FtController,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll get_status() until mode returns to MONITORING."""
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        status = controller.get_status()
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
        f"last status: {controller.get_status()}"
    )


async def wait_for_training_stable(
    controller: FtController,
    mini_wandb: MiniWandb,
    n_iterations: int = 10,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> None:
    """Poll MiniWandb for N consecutive successful iterations."""
    baseline = get_iteration_count(mini_wandb=mini_wandb)
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        current = get_iteration_count(mini_wandb=mini_wandb)
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
    current = get_iteration_count(mini_wandb=mini_wandb)
    raise TimeoutError(
        f"Training did not stabilize: need {n_iterations} iterations, "
        f"got {current - baseline} in {timeout}s"
    )


def get_iteration_count(mini_wandb: MiniWandb) -> int:
    """Query current iteration from MiniWandb."""
    value = mini_wandb.latest(metric_name="iteration")
    if value is None:
        return 0
    return int(value)


async def wait_for_recovery_phase(
    controller: FtController,
    phase: RecoveryPhase,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> ControllerStatus:
    """Poll get_status() until recovery_phase matches."""
    deadline = time.monotonic() + timeout
    poll_count = 0
    while time.monotonic() < deadline:
        status = controller.get_status()
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
        f"last status: {controller.get_status()}"
    )


async def wait_for_mode_transition(
    controller: FtController,
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
        status = controller.get_status()
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
            f"last status: {controller.get_status()}"
        )

    while time.monotonic() < deadline:
        status = controller.get_status()
        if status.mode == target_mode:
            return status
        await asyncio.sleep(poll_interval)

    raise TimeoutError(
        f"Mode did not return to '{target_mode}' within {timeout}s, "
        f"last status: {controller.get_status()}"
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
