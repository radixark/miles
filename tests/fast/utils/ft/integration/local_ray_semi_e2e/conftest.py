from __future__ import annotations

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
import ray

from miles.utils.ft.agents.collectors.stub import StubCollector
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.metric_names import AGENT_HEARTBEAT
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.platform.config import FtControllerConfig
from miles.utils.ft.platform.ray_wrappers.controller_actor import FtControllerActor
from miles.utils.ft.platform.ray_wrappers.node_agent_actor import FtNodeAgentActor
from miles.utils.ft.protocols.platform import (
    JobStatus,
    ft_controller_actor_name,
    ft_node_agent_actor_name,
)

from tests.fast.utils.ft.helpers.controller_fakes import FakeNodeManager
from tests.fast.utils.ft.helpers.diagnostic_fakes import StubDiagnostic
from tests.fast.utils.ft.helpers.fault_injection import LocalRayFaultInjector
from tests.fast.utils.ft.helpers.training_simulator import (
    CollectorStateActor,
    RemoteControlledCollector,
    RemoteControlledTrainingJob,
    TrainingStateActor,
    TrainingWorkerActor,
)
from tests.fast.utils.ft.integration.conftest import (
    _kill_named_actor,
    poll_for_run_id,
)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class NodeSpec:
    node_id: str
    num_ranks: int = 1
    diagnostic_pass: bool = True
    diagnostic_types: list[str] = field(default_factory=lambda: ["gpu"])
    use_remote_collector: bool = False


@dataclass
class E2EEnv:
    controller: ray.actor.ActorHandle
    state_actor: ray.actor.ActorHandle
    injector: LocalRayFaultInjector
    ft_id: str
    node_agents: dict[str, ray.actor.ActorHandle] = field(default_factory=dict)
    workers: list[ray.actor.ActorHandle] = field(default_factory=list)
    collector_states: dict[str, ray.actor.ActorHandle] = field(default_factory=dict)
    _cleanup_names: list[str] = field(default_factory=list)
    _cleanup_handles: list[ray.actor.ActorHandle] = field(default_factory=list)

    def set_collector_metrics(
        self, node_id: str, metrics: list[GaugeSample],
    ) -> None:
        state = self.collector_states.get(node_id)
        if state is None:
            raise KeyError(f"No collector state for node {node_id}")
        ray.get(state.set_metrics.remote(metrics), timeout=5)

    def cleanup(self) -> None:
        for worker in self.workers:
            try:
                ray.get(worker.stop.remote(), timeout=5)
            except Exception:
                pass

        for agent in self.node_agents.values():
            try:
                ray.get(agent.stop.remote(), timeout=5)
            except Exception:
                pass

        try:
            ray.get(self.controller.shutdown.remote(), timeout=10)
        except Exception:
            pass

        for name in self._cleanup_names:
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except (ValueError, Exception):
                pass

        for handle in self._cleanup_handles:
            try:
                ray.kill(handle, no_restart=True)
            except Exception:
                pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_FAST_TICK = 0.1
_FAST_SCRAPE = 0.5
_FAST_STEP = 0.1
_SLOW_STEP = 2.0


def _wait_for_iteration_advancing(
    controller: ray.actor.ActorHandle,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = ray.get(controller.get_status.remote(), timeout=5)
        if status.latest_iteration is not None and status.latest_iteration > 0:
            return
        time.sleep(0.3)
    raise TimeoutError("Worker iteration metrics not visible on controller within timeout")


def _wait_for_metrics_scraped(
    controller: ray.actor.ActorHandle,
    scrape_interval: float = _FAST_SCRAPE,
    cycles: int = 3,
) -> None:
    """Wait enough time for MiniPrometheus to scrape several cycles."""
    time.sleep(scrape_interval * cycles)


class _FastHangDetector(BaseFaultDetector):
    """HangDetector with sub-minute timeout for fast testing."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self._timeout = timedelta(seconds=timeout_seconds)

    def evaluate(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision(action=ActionType.NONE, reason="not running")

        df = ctx.metric_store.changes(
            AGENT_HEARTBEAT,
            window=self._timeout,
            label_filters={"rank": "0"},
        )
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="no iteration data")

        if df["value"][0] == 0:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"iteration stalled for {self._timeout.total_seconds()}s",
                trigger=TriggerType.HANG,
            )
        return Decision(action=ActionType.NONE, reason="progressing")


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------


def _build_e2e_env(
    *,
    ft_id: str = "e2e",
    nodes: list[NodeSpec] | None = None,
    detectors: list[BaseFaultDetector] | None = None,
    tick_interval: float = _FAST_TICK,
    scrape_interval_seconds: float = 10.0,
    step_interval: float = _FAST_STEP,
    recovery_cooldown: SlidingWindowThrottle | None = None,
    registration_grace_ticks: int | None = None,
    notifier_override: Any = None,
    wait_for_iteration: bool = True,
    max_simultaneous_bad_nodes: int | None = None,
    recovery_timeout_seconds: int | None = None,
    monitoring_timeout_seconds: int | None = None,
    monitoring_success_iterations: int | None = None,
) -> E2EEnv:
    if nodes is None:
        nodes = [NodeSpec(node_id=f"{ft_id}-node-0")]

    if detectors is None:
        detectors = [TrainingCrashDetector()]

    state_actor = TrainingStateActor.remote()
    training_job = RemoteControlledTrainingJob(state_actor=state_actor)

    controller_kwargs: dict[str, Any] = dict(
        config=FtControllerConfig(
            platform="stub",
            tick_interval=tick_interval,
            ft_id=ft_id,
            scrape_interval_seconds=scrape_interval_seconds,
        ),
        training_job_override=training_job,
        node_manager_override=FakeNodeManager(),
        notifier_override=notifier_override,
        detectors_override=detectors,
        start_exporter=True,
    )
    if recovery_cooldown is not None:
        controller_kwargs["recovery_cooldown_override"] = recovery_cooldown
    if registration_grace_ticks is not None:
        controller_kwargs["registration_grace_ticks_override"] = registration_grace_ticks
    if max_simultaneous_bad_nodes is not None:
        controller_kwargs["max_simultaneous_bad_nodes_override"] = max_simultaneous_bad_nodes
    if recovery_timeout_seconds is not None:
        controller_kwargs["recovery_timeout_seconds_override"] = recovery_timeout_seconds
    if monitoring_timeout_seconds is not None:
        controller_kwargs["monitoring_timeout_seconds_override"] = monitoring_timeout_seconds
    if monitoring_success_iterations is not None:
        controller_kwargs["monitoring_success_iterations_override"] = monitoring_success_iterations

    controller_name = ft_controller_actor_name(ft_id)
    controller = FtControllerActor.options(name=controller_name).remote(**controller_kwargs)
    controller.submit_and_run.remote()
    run_id = poll_for_run_id(controller)

    env = E2EEnv(
        controller=controller,
        state_actor=state_actor,
        injector=LocalRayFaultInjector(state_actor=state_actor),
        ft_id=ft_id,
    )
    env._cleanup_names.append(controller_name)
    env._cleanup_handles.append(state_actor)

    # Step: start node agents
    rank_offset = 0
    for node_spec in nodes:
        collector_state: ray.actor.ActorHandle | None = None
        if node_spec.use_remote_collector:
            collector_state = CollectorStateActor.remote()
            env.collector_states[node_spec.node_id] = collector_state
            env._cleanup_handles.append(collector_state)
            collectors: list[Any] = [RemoteControlledCollector(
                state_actor=collector_state,
                collect_interval=0.3,
            )]
        else:
            collectors = [StubCollector()]

        diagnostics = [
            StubDiagnostic(
                passed=node_spec.diagnostic_pass,
                diagnostic_type=dt,
            )
            for dt in node_spec.diagnostic_types
        ]

        agent_name = ft_node_agent_actor_name(ft_id, node_spec.node_id)
        node_agent = FtNodeAgentActor.options(name=agent_name).remote(
            node_id=node_spec.node_id,
            ft_id=ft_id,
            collect_interval_seconds=0.3,
            collectors_override=collectors,
            diagnostics_override=diagnostics,
        )
        ray.get(node_agent.start.remote(), timeout=10)
        env.node_agents[node_spec.node_id] = node_agent
        env._cleanup_names.append(agent_name)

        # Step: start training workers for this node
        for local_rank in range(node_spec.num_ranks):
            global_rank = rank_offset + local_rank
            total_world_size = sum(n.num_ranks for n in nodes)
            worker = TrainingWorkerActor.remote(
                state_actor=state_actor,
                ft_id=ft_id,
                rank=global_rank,
                world_size=total_world_size,
                node_id=node_spec.node_id,
                step_interval=step_interval,
            )
            ray.get(worker.start.remote(), timeout=10)
            env.workers.append(worker)
            env._cleanup_handles.append(worker)

        rank_offset += node_spec.num_ranks

    if wait_for_iteration:
        _wait_for_iteration_advancing(controller)

    return env


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def _cleanup_default_controller(local_ray: None) -> Generator[None, None, None]:
    _kill_named_actor(ft_controller_actor_name(""))
    yield
    _kill_named_actor(ft_controller_actor_name(""))


@pytest.fixture
def e2e_env(local_ray: None) -> Generator[E2EEnv, None, None]:
    """1 node + 1 rank, TrainingCrashDetector, StubCollector/StubDiagnostic."""
    env = _build_e2e_env(
        ft_id="e2e",
        nodes=[NodeSpec(node_id="e2e-node-0")],
        detectors=[TrainingCrashDetector()],
    )
    yield env
    env.cleanup()


@pytest.fixture
def e2e_multi_node_env(local_ray: None) -> Generator[E2EEnv, None, None]:
    """2 nodes + 4 ranks (2 per node), both diagnostics pass by default."""
    env = _build_e2e_env(
        ft_id="e2emn",
        nodes=[
            NodeSpec(node_id="e2emn-node-0", num_ranks=2),
            NodeSpec(node_id="e2emn-node-1", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )
    yield env
    env.cleanup()


@pytest.fixture
def e2e_full_detector_env(local_ray: None) -> Generator[E2EEnv, None, None]:
    """Full detector chain + RemoteControlledCollector + fast scrape."""
    env = _build_e2e_env(
        ft_id="e2efd",
        nodes=[NodeSpec(
            node_id="e2efd-node-0",
            use_remote_collector=True,
        )],
        detectors=build_detector_chain(),
        scrape_interval_seconds=_FAST_SCRAPE,
    )
    yield env
    env.cleanup()


@pytest.fixture
def e2e_hang_env(local_ray: None) -> Generator[E2EEnv, None, None]:
    """FastHangDetector (3s timeout) + TrainingWorkerActor."""
    env = _build_e2e_env(
        ft_id="e2ehng",
        nodes=[NodeSpec(node_id="e2ehng-node-0")],
        detectors=[_FastHangDetector(timeout_seconds=3.0)],
    )
    yield env
    env.cleanup()


@pytest.fixture
def make_e2e_env(local_ray: None) -> Generator[Callable[..., E2EEnv], None, None]:
    """Factory fixture for one-off E2E configurations."""
    created: list[E2EEnv] = []

    def _factory(**kwargs: Any) -> E2EEnv:
        env = _build_e2e_env(**kwargs)
        created.append(env)
        return env

    yield _factory

    for env in created:
        env.cleanup()
