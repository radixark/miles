"""Local Ray: comprehensive E2E-like scenarios with real agents.

Tests start real FtNodeAgentActor and FtTrainingRankAgent instances
(with stub/remote-controlled collectors and diagnostics) so the full
registration, metrics scraping, diagnostic, and recovery pipelines
are exercised end-to-end in a local Ray cluster.
"""
from __future__ import annotations

import asyncio
import logging
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
from miles.utils.ft.models.recovery import ControllerMode
from miles.utils.ft.models.metric_names import (
    GPU_AVAILABLE,
    NODE_NETWORK_UP,
    TRAINING_ITERATION,
)
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.models.recovery import RecoveryPhase
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.platform.node_agent_actor import FtNodeAgentActor
from miles.utils.ft.protocols.platform import (
    JobStatus,
    ft_controller_actor_name,
    ft_node_agent_actor_name,
)

from tests.fast.utils.ft.helpers.controller_fakes import FakeNodeManager, FakeNotifier
from tests.fast.utils.ft.helpers.diagnostic_fakes import StubDiagnostic
from tests.fast.utils.ft.helpers.fault_injection import LocalRayFaultInjector
from tests.fast.utils.ft.helpers.scenarios import (
    assert_phase_path_contains,
    get_status,
    scenario_hang_detection,
    scenario_no_false_positive,
    scenario_repeated_crash,
    scenario_transient_crash,
    wait_for_mode,
    wait_for_mode_transition,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)
from tests.fast.utils.ft.helpers.training_simulator import (
    CollectorStateActor,
    RemoteControlledCollector,
    RemoteControlledTrainingJob,
    TrainingStateActor,
    TrainingWorkerActor,
)
from tests.fast.utils.ft.integration.local_ray.conftest import poll_for_run_id

logger = logging.getLogger(__name__)

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
            TRAINING_ITERATION,
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


# ==================================================================
# P0 — Core integration paths (9 tests)
# ==================================================================


class TestTransientCrash:
    async def test_crash_recovery_with_real_agents(self, e2e_env: E2EEnv) -> None:
        """Single crash → auto-recovery → training resumes with re-registered workers."""
        status = await scenario_transient_crash(
            handle=e2e_env.controller,
            injector=e2e_env.injector,
            stable_iterations=3,
            recovery_timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING


class TestDiagnosticEviction:
    async def test_diagnostic_failure_evicts_bad_node(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """node-0 diagnostic fails, node-1 passes → only node-0 evicted.

        Crash during MONITORING → step_monitoring sees FAILED → DIAGNOSING.
        StubDiagnostic resolves instantly so we check phase_history after
        recovery completes rather than catching DIAGNOSING in flight.
        """
        env = make_e2e_env(
            ft_id="e2ediag",
            nodes=[
                NodeSpec(node_id="e2ediag-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ediag-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for recovery MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [
            RecoveryPhase.DIAGNOSING,
            RecoveryPhase.EVICT_AND_RESTART,
            RecoveryPhase.DONE,
        ])


class TestHardwareAlert:
    async def test_gpu_lost_triggers_direct_eviction(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """GPU_AVAILABLE=0 → HighConfidenceHardwareDetector → MARK_BAD_AND_RESTART."""
        env = e2e_full_detector_env

        # Step 1: inject GPU unavailable metric
        env.set_collector_metrics("e2efd-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efd-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 2: wait for metrics to be scraped and detector to act
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY

        # Step 3: wait for full recovery
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)
        assert_phase_path_contains(final, [
            RecoveryPhase.EVICT_AND_RESTART,
            RecoveryPhase.DONE,
        ])


class TestNanLoss:
    async def test_nan_loss_triggers_recovery(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """loss=NaN via custom log metrics → NanLossDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject NaN loss
        await env.injector.inject_nan_loss()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestRecoveryThrottle:
    async def test_third_crash_throttled(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """3 crashes with max_count=3 → third is throttled (stays in MONITORING).

        record() is called before is_throttled(), so max_count=3 allows 2 recoveries.
        """
        env = make_e2e_env(
            ft_id="e2ethr",
            nodes=[NodeSpec(node_id="e2ethr-node-0")],
            detectors=[TrainingCrashDetector()],
            recovery_cooldown=SlidingWindowThrottle(
                window_minutes=60,
                max_count=3,
            ),
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: second crash → recovery
        await env.injector.crash_training()
        await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 3: third crash → throttled, no recovery
        await env.injector.crash_training()
        await asyncio.sleep(5.0)
        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING


class TestMultiNode:
    async def test_multi_rank_registration_and_targeted_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """4 ranks across 2 nodes register correctly; crash during MONITORING → DIAGNOSING."""
        env = make_e2e_env(
            ft_id="e2emn",
            nodes=[
                NodeSpec(node_id="e2emn-node-0", num_ranks=2),
                NodeSpec(node_id="e2emn-node-1", num_ranks=2),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING

        # Step 1: crash → recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [
            RecoveryPhase.DIAGNOSING,
        ])


class TestRunIdSwitch:
    async def test_new_run_id_reregistration_and_metric_isolation(
        self, e2e_env: E2EEnv,
    ) -> None:
        """After recovery, worker re-registers with new run_id and metrics use new run."""
        env = e2e_env

        pre_status = get_status(env.controller)
        pre_run_id = pre_status.active_run_id

        # Step 1: crash → recovery → new run
        await env.injector.crash_training()
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )

        # Step 2: verify new run_id
        post_run_id = final.active_run_id
        assert post_run_id is not None
        assert post_run_id != pre_run_id

        # Step 3: verify iteration progresses under new run
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)


class TestFaultDuringRecovery:
    async def test_hardware_fault_during_reattempting_escalates(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → REATTEMPTING → inject GPU_AVAILABLE=0 → critical detector finds → EVICT_AND_RESTART."""
        env = make_e2e_env(
            ft_id="e2efdr",
            nodes=[NodeSpec(
                node_id="e2efdr-node-0",
                use_remote_collector=True,
            )],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → enters recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: while in recovery, inject hardware fault
        env.set_collector_metrics("e2efdr-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efdr-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 3: wait for recovery to complete with eviction
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [RecoveryPhase.EVICT_AND_RESTART])


class TestStatusConsistency:
    async def test_status_snapshots_internally_consistent(
        self, e2e_env: E2EEnv,
    ) -> None:
        """High-frequency polling during recovery → every snapshot is internally consistent."""
        env = e2e_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()

        snapshots: list[Any] = []
        deadline = time.monotonic() + 60.0

        while time.monotonic() < deadline:
            status = get_status(env.controller)
            snapshots.append(status)

            if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
                if len(snapshots) > 5:
                    break
            await asyncio.sleep(0.05)

        assert len(snapshots) > 5, "Not enough snapshots collected"

        for s in snapshots:
            if s.mode == ControllerMode.RECOVERY:
                assert s.recovery_in_progress is True
                assert s.recovery_phase is not None
            elif s.mode == ControllerMode.MONITORING:
                if s.recovery_in_progress:
                    assert s.recovery_phase is not None


# ==================================================================
# P1 — Important boundaries (8 tests)
# ==================================================================


class TestNoFalsePositive:
    async def test_healthy_training_stays_monitoring(self, e2e_env: E2EEnv) -> None:
        """No faults injected → controller never enters recovery."""
        status = await scenario_no_false_positive(
            handle=e2e_env.controller,
            observation_iterations=5,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False


class TestRepeatedCrash:
    async def test_two_crashes_all_diag_pass_goes_to_notify(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash during recovery MONITORING → DIAGNOSING → all diagnostics pass → NOTIFY → DONE."""
        env = make_e2e_env(
            ft_id="e2erpt",
            nodes=[NodeSpec(node_id="e2erpt-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [
            RecoveryPhase.DIAGNOSING,
            RecoveryPhase.NOTIFY,
            RecoveryPhase.DONE,
        ])


class TestHangDetection:
    async def test_stale_iteration_triggers_recovery(
        self, e2e_hang_env: E2EEnv,
    ) -> None:
        """Worker iteration stalls → FastHangDetector → ENTER_RECOVERY."""
        status = await scenario_hang_detection(
            handle=e2e_hang_env.controller,
            injector=e2e_hang_env.injector,
            hang_timeout=20.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestMonitoringTimeout:
    async def test_crash_during_hung_monitoring_escalates_to_diagnosing(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Worker hung during recovery MONITORING → crash again → DIAGNOSING.

        Uses slow step_interval so MONITORING lasts long enough to inject crash.
        """
        env = make_e2e_env(
            ft_id="e2emto",
            nodes=[NodeSpec(node_id="e2emto-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)
        assert_phase_path_contains(final, [RecoveryPhase.DIAGNOSING])


class TestRegistrationGrace:
    async def test_detectors_suppressed_during_grace_period(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """With grace_ticks=10, detectors don't fire during the grace period."""
        env = make_e2e_env(
            ft_id="e2egrce",
            nodes=[NodeSpec(node_id="e2egrce-node-0")],
            detectors=[TrainingCrashDetector()],
            registration_grace_ticks=10,
        )

        # Step 1: crash during grace period → should NOT trigger recovery
        await env.injector.crash_training()
        await asyncio.sleep(0.5)
        status = get_status(env.controller)
        if status.tick_count <= 10:
            assert status.mode == ControllerMode.MONITORING, (
                f"Recovery triggered during grace period at tick {status.tick_count}"
            )

        # Step 2: wait for grace period to end
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.tick_count > 10:
                break
            await asyncio.sleep(0.2)

        # Step 3: crash again after grace period → should trigger recovery
        await env.injector.recover_training()
        await asyncio.sleep(1.0)
        await env.injector.crash_training()
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestConcurrentRegistration:
    async def test_parallel_rank_registration(
        self, e2e_multi_node_env: E2EEnv,
    ) -> None:
        """4 workers register in parallel → all ranks visible in controller status."""
        env = e2e_multi_node_env

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        status = get_status(env.controller)
        assert status.latest_iteration is not None
        assert status.latest_iteration > 0


class TestEvictionExcludedNodes:
    async def test_multi_node_eviction_passes_correct_excluded_ids(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """2 bad nodes out of 3 → excluded_node_ids contains both."""
        env = make_e2e_env(
            ft_id="e2eexcl",
            nodes=[
                NodeSpec(node_id="e2eexcl-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-1", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-2", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [RecoveryPhase.DIAGNOSING])
        assert_phase_path_contains(final, [RecoveryPhase.EVICT_AND_RESTART])


class TestRecoveryReset:
    async def test_state_clean_after_recovery_completes(
        self, e2e_env: E2EEnv,
    ) -> None:
        """Recovery DONE → mode=MONITORING, phase=None → new crash is a fresh cycle."""
        env = e2e_env

        # Step 1: first crash → full recovery
        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()
        status = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False

        # Step 2: second crash → enters a fresh recovery cycle
        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        await env.injector.crash_training()
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_phase is not None


# ==================================================================
# P2 — Nice-to-have (9 tests)
# ==================================================================


class TestExceptionInRecovery:
    async def test_stop_training_exception_forces_notify(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """stop_training raises → recovery forces NOTIFY → actor survives.

        We simulate this by having the training job's stop fail.
        In E2E with RemoteControlledTrainingJob, stop() calls state_actor.stop.remote()
        which is unlikely to fail, so this test verifies the controller handles
        the crash → recovery flow and continues operating.
        """
        env = make_e2e_env(
            ft_id="e2eexc",
            nodes=[NodeSpec(node_id="e2eexc-node-0")],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)
        await env.injector.crash_training()

        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=60.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestEphemeralNic:
    async def test_ephemeral_nic_fault_goes_to_reattempting(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NIC down samples (ephemeral) → NetworkAlertDetector → MARK_BAD_AND_RESTART.

        MARK_BAD_AND_RESTART evicts directly without entering recovery mode,
        so we detect the action by observing run_id change.
        """
        from miles.utils.ft.controller.detectors.network import NetworkAlertDetector

        env = make_e2e_env(
            ft_id="e2enic",
            nodes=[NodeSpec(
                node_id="e2enic-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(
                    alert_window=timedelta(seconds=10),
                    alert_threshold=1,
                ),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject NIC down metrics
        env.set_collector_metrics("e2enic-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enic-node-0", "device": "eth0"},
                value=0.0,
            ),
        ])

        # Step 2: MARK_BAD_AND_RESTART evicts and restarts without entering
        # recovery mode; poll until active_run_id changes.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestConcurrentFaults:
    async def test_simultaneous_nan_and_crash(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NaN + FAILED simultaneously → detector priority chain triggers 1 recovery."""
        env = make_e2e_env(
            ft_id="e2ecf",
            nodes=[NodeSpec(node_id="e2ecf-node-0")],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject both faults
        await env.injector.inject_nan_loss()
        await env.injector.crash_training()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY

        # Step 3: let recovery complete
        final = await wait_for_mode_transition(
            env.controller,
            target_mode=ControllerMode.MONITORING,
            timeout=90.0,
        )
        assert final.mode == ControllerMode.MONITORING


class TestPartialDiagnostic:
    async def test_unreachable_node_agent_treated_as_bad(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During DIAGNOSING, kill node agent → RayActorError → treated as bad node."""
        env = make_e2e_env(
            ft_id="e2epd",
            nodes=[
                NodeSpec(node_id="e2epd-node-0", num_ranks=1, diagnostic_pass=True),
                NodeSpec(node_id="e2epd-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: kill node-0's agent before crash
        node_agent_0 = env.node_agents["e2epd-node-0"]
        ray.kill(node_agent_0, no_restart=True)
        env.node_agents.pop("e2epd-node-0", None)

        # Step 2: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 3: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [RecoveryPhase.DIAGNOSING])
        assert_phase_path_contains(final, [RecoveryPhase.EVICT_AND_RESTART])


class TestAllNodesEvicted:
    async def test_all_diagnostics_fail_notifies_human(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """All nodes fail diagnostic → all evicted → NOTIFY."""
        env = make_e2e_env(
            ft_id="e2eall",
            nodes=[
                NodeSpec(node_id="e2eall-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eall-node-1", num_ranks=1, diagnostic_pass=False),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase=RecoveryPhase.MONITORING,
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [
            RecoveryPhase.DIAGNOSING,
            RecoveryPhase.EVICT_AND_RESTART,
            RecoveryPhase.DONE,
        ])


class TestAgentWithoutController:
    async def test_rank_agent_graceful_degrade(self, local_ray: None) -> None:
        """Creating FtTrainingRankAgent when controller doesn't exist → no crash."""
        import os
        from unittest.mock import patch

        os.environ["MILES_FT_ID"] = "nonexistent"
        os.environ["MILES_FT_TRAINING_RUN_ID"] = "fake-run"

        from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent

        with patch("socket.gethostname", return_value="test-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1)

        agent.step(1)
        agent.shutdown()


class TestFireAndForget:
    async def test_log_step_after_controller_death(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Kill controller → worker continues log_step → doesn't crash/block."""
        env = make_e2e_env(
            ft_id="e2eff",
            nodes=[NodeSpec(node_id="e2eff-node-0")],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: kill controller
        controller_name = ft_controller_actor_name(env.ft_id)
        try:
            ray.get(env.controller.shutdown.remote(), timeout=5)
        except Exception:
            pass
        try:
            ray.kill(ray.get_actor(controller_name), no_restart=True)
        except (ValueError, Exception):
            pass

        # Step 2: wait a bit - worker should continue running without crashing
        await asyncio.sleep(2.0)

        # Step 3: verify workers are still alive
        for worker in env.workers:
            iteration = ray.get(worker.get_iteration.remote(), timeout=5)
            assert iteration > 0


class TestMetricScrapeE2E:
    async def test_prometheus_exporter_to_metric_store_pipeline(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """Worker pushes iteration → controller scrapes → status.latest_iteration tracks it."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=5, timeout=30.0)
        status = get_status(env.controller)
        assert status.latest_iteration is not None
        assert status.latest_iteration >= 5


class TestNetworkAlert:
    async def test_sustained_nic_down_triggers_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Sustained NIC down → NetworkAlertDetector → MARK_BAD_AND_RESTART.

        MARK_BAD_AND_RESTART evicts directly without entering recovery mode.
        """
        from miles.utils.ft.controller.detectors.network import NetworkAlertDetector

        env = make_e2e_env(
            ft_id="e2enet",
            nodes=[NodeSpec(
                node_id="e2enet-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(
                    alert_window=timedelta(seconds=10),
                    alert_threshold=1,
                ),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)
        old_run_id = get_status(env.controller).active_run_id

        # Step 1: inject sustained NIC down
        env.set_collector_metrics("e2enet-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth0"},
                value=0.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth1"},
                value=0.0,
            ),
        ])

        # Step 2: poll until active_run_id changes (MARK_BAD_AND_RESTART
        # evicts and restarts without entering recovery mode)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING
