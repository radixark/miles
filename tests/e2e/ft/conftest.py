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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

import pytest
import ray
from ray.job_submission import JobSubmissionClient

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.e2e.fault_injector import FaultInjectorActor, deploy_fault_injector
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from miles.utils.ft.platform.protocols import JobStatus
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
# Session-scoped: FT system (Controller + metric store)
# ---------------------------------------------------------------------------


@dataclass
class FtSystem:
    controller: FtController
    controller_handle: Any
    metric_store: MiniPrometheus
    mini_wandb: MiniWandb
    training_job: RayTrainingJob
    node_manager: K8sNodeManager
    _controller_task: asyncio.Task[None] | None = field(default=None, repr=False)

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._controller_task = loop.create_task(self.controller.run())

    async def shutdown(self) -> None:
        await self.controller.shutdown()
        if self._controller_task is not None:
            try:
                await asyncio.wait_for(self._controller_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._controller_task.cancel()


@pytest.fixture(scope="session")
def ft_system(ray_cluster: None, ray_address: str) -> Generator[FtSystem, None, None]:
    """Start FT Controller with mini metric store."""
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
    node_manager = K8sNodeManager()
    training_job = RayTrainingJob(
        client=JobSubmissionClient(address=ray_address),
        entrypoint=entrypoint,
    )

    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=mini_prom,
        mini_wandb=mini_wandb,
        detectors=build_detector_chain(),
        tick_interval=5.0,
        controller_exporter=controller_exporter,
        scrape_target_manager=mini_prom,
    )

    system = FtSystem(
        controller=controller,
        controller_handle=controller,
        metric_store=mini_prom,
        mini_wandb=mini_wandb,
        training_job=training_job,
        node_manager=node_manager,
    )

    yield system


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
def target_node(ray_cluster: None) -> str:
    """Pick the first alive non-head node from the cluster."""
    nodes = ray.nodes()
    alive_nodes = [
        n for n in nodes
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
    ]
    if not alive_nodes:
        pytest.skip("No alive GPU nodes found")
    return alive_nodes[0]["NodeID"]


# ---------------------------------------------------------------------------
# Helper functions for E2E assertions
# ---------------------------------------------------------------------------


def wait_for_condition(
    check_fn: Any,
    timeout: float,
    interval: float = 5.0,
    description: str = "condition",
) -> Any:
    """Poll check_fn until it returns a truthy value or timeout."""
    deadline = time.monotonic() + timeout
    last_result = None
    while time.monotonic() < deadline:
        last_result = check_fn()
        if last_result:
            return last_result
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {description} after {timeout}s")


async def wait_for_recovery_complete(
    controller: FtController,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    """Poll get_status() until mode returns to 'monitoring'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = controller.get_status()
        if status["mode"] == "monitoring":
            return status
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
    while time.monotonic() < deadline:
        current = get_iteration_count(mini_wandb=mini_wandb)
        if current - baseline >= n_iterations:
            return
        await asyncio.sleep(poll_interval)
    current = get_iteration_count(mini_wandb=mini_wandb)
    raise TimeoutError(
        f"Training did not stabilize: need {n_iterations} iterations, "
        f"got {current - baseline} in {timeout}s"
    )


def get_iteration_count(mini_wandb: MiniWandb) -> int:
    """Query current iteration from MiniWandb."""
    value = mini_wandb.latest(metric_name="iteration", rank=0)
    if value is None:
        return 0
    return int(value)


async def wait_for_recovery_phase(
    controller: FtController,
    phase: str,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    """Poll get_status() until recovery_phase matches."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = controller.get_status()
        if status.get("recovery_phase") == phase:
            return status
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Did not reach recovery phase '{phase}' within {timeout}s, "
        f"last status: {controller.get_status()}"
    )
