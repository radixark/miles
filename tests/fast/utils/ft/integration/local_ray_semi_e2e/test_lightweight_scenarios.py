"""Local Ray: E2E-like shared scenarios — transient crash, no false positive, hang."""
from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
from typing import Any

import pytest
import ray
from prometheus_client import Gauge

from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models.recovery import ControllerMode
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.metric_names import AGENT_HEARTBEAT
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import JobStatus, ft_controller_actor_name

from tests.fast.utils.ft.helpers.fault_injection import LocalRayFaultInjector
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    scenario_hang_detection,
    scenario_no_false_positive,
    scenario_repeated_crash,
    scenario_transient_crash,
)
from tests.fast.utils.ft.helpers.training_simulator import (
    RemoteControlledTrainingJob,
    TrainingStateActor,
)
from tests.fast.utils.ft.integration.conftest import poll_for_run_id


@pytest.fixture
def simulated_env(
    local_ray: None,
) -> Generator[tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector], None, None]:
    """Create a controller backed by RemoteControlledTrainingJob + TrainingStateActor.

    Yields (controller_handle, state_actor, fault_injector).
    """
    state_actor = TrainingStateActor.remote()
    training_job = RemoteControlledTrainingJob(state_actor=state_actor)

    actor_name = ft_controller_actor_name("")
    controller = FtControllerActor.options(name=actor_name).remote(
        config=FtControllerConfig(platform="stub", tick_interval=0.1),
        training_job_override=training_job,
        node_manager_override=_make_fake_node_manager(),
        notifier_override=None,
        detectors_override=[TrainingCrashDetector()],
    )

    controller.submit_and_run.remote()
    run_id = poll_for_run_id(controller)

    ray.get(controller.register_training_rank.remote(
        run_id=run_id, rank=0, world_size=1,
        node_id="sim-node-0", exporter_address="http://sim-node-0:9090",
        pid=1000,
    ), timeout=5)

    injector = LocalRayFaultInjector(state_actor=state_actor)

    yield controller, state_actor, injector

    try:
        ray.get(controller.shutdown.remote(), timeout=10)
    except Exception:
        pass
    try:
        ray.kill(ray.get_actor(actor_name), no_restart=True)
    except ValueError:
        pass
    try:
        ray.kill(state_actor, no_restart=True)
    except Exception:
        pass


def _make_fake_node_manager() -> Any:
    from tests.fast.utils.ft.helpers.controller_fakes import FakeNodeManager
    return FakeNodeManager()


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


@pytest.fixture
def hang_simulated_env(
    local_ray: None,
) -> Generator[tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector], None, None]:
    state_actor = TrainingStateActor.remote()
    training_job = RemoteControlledTrainingJob(state_actor=state_actor)

    exporter = PrometheusExporter()
    iteration_gauge = Gauge(
        AGENT_HEARTBEAT,
        "iteration gauge for hang test",
        labelnames=["rank", "node_id"],
        registry=exporter.registry,
    )
    iteration_gauge.labels(rank="0", node_id="hang-node").set(42.0)

    actor_name = ft_controller_actor_name("hang")
    controller = FtControllerActor.options(name=actor_name).remote(
        config=FtControllerConfig(platform="stub", tick_interval=0.1, ft_id="hang"),
        training_job_override=training_job,
        node_manager_override=_make_fake_node_manager(),
        notifier_override=None,
        detectors_override=[_FastHangDetector(timeout_seconds=3.0)],
    )

    controller.submit_and_run.remote()
    run_id = poll_for_run_id(controller)

    ray.get(controller.register_training_rank.remote(
        run_id=run_id, rank=0, world_size=1,
        node_id="hang-node", exporter_address=exporter.get_address(),
        pid=1000,
    ), timeout=5)

    injector = LocalRayFaultInjector(state_actor=state_actor)

    yield controller, state_actor, injector

    exporter.shutdown()
    try:
        ray.get(controller.shutdown.remote(), timeout=10)
    except Exception:
        pass
    try:
        ray.kill(ray.get_actor(actor_name), no_restart=True)
    except ValueError:
        pass
    try:
        ray.kill(state_actor, no_restart=True)
    except Exception:
        pass


class TestTransientCrash:
    async def test_crash_triggers_recovery_then_returns_to_monitoring(
        self,
        simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, state_actor, injector = simulated_env

        status = await scenario_transient_crash(
            handle=controller,
            injector=injector,
            stable_iterations=0,
            recovery_timeout=30.0,
        )

        assert status.mode == ControllerMode.MONITORING


class TestNoFalsePositive:
    async def test_healthy_training_stays_in_monitoring(
        self,
        simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, state_actor, injector = simulated_env

        status = await scenario_no_false_positive(
            handle=controller,
            observation_ticks=20,
            poll_interval=0.2,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False


class TestRepeatedCrash:
    async def test_two_crashes_escalate_to_diagnosing(
        self,
        simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, _state_actor, injector = simulated_env

        await scenario_repeated_crash(
            handle=controller,
            injector=injector,
            stable_iterations=0,
            recovery_timeout=30.0,
        )


class TestHangDetection:
    async def test_stale_iteration_triggers_hang_recovery(
        self,
        hang_simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, state_actor, injector = hang_simulated_env

        status = await scenario_hang_detection(
            handle=controller,
            injector=injector,
            hang_timeout=20.0,
        )

        assert status.mode == ControllerMode.RECOVERY
