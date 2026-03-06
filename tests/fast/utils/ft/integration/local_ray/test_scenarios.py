"""Local Ray: E2E-like shared scenarios — transient crash, no false positive."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Generator
from typing import Any

import pytest
import ray

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

from tests.fast.utils.ft.helpers.fault_injection import LocalRayFaultInjector
from tests.fast.utils.ft.helpers.scenarios import (
    assert_phase_path_contains,
    get_status,
    scenario_no_false_positive,
    scenario_transient_crash,
)
from tests.fast.utils.ft.helpers.training_simulator import (
    RemoteControlledTrainingJob,
    TrainingStateActor,
)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


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

    from tests.fast.utils.ft.integration.local_ray.conftest import poll_for_run_id
    run_id = poll_for_run_id(controller)

    ray.get(controller.register_training_rank.remote(
        run_id=run_id, rank=0, world_size=1,
        node_id="sim-node-0", exporter_address="http://sim-node-0:9090",
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
    async def test_two_crashes_both_trigger_recovery(
        self,
        simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, state_actor, injector = simulated_env

        await injector.crash_training()

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            s = get_status(controller)
            if s.mode == ControllerMode.RECOVERY:
                break
            await asyncio.sleep(0.2)

        deadline2 = time.monotonic() + 30.0
        while time.monotonic() < deadline2:
            s = get_status(controller)
            if s.mode == ControllerMode.MONITORING and s.tick_count > 5:
                break
            await asyncio.sleep(0.3)

        await injector.crash_training()

        deadline3 = time.monotonic() + 15.0
        entered_recovery_again = False
        while time.monotonic() < deadline3:
            s = get_status(controller)
            if s.mode == ControllerMode.RECOVERY:
                entered_recovery_again = True
                break
            await asyncio.sleep(0.2)

        assert entered_recovery_again, "Second crash did not trigger recovery"
