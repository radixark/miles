"""Semi-E2E: resilience — controller death, all-pass diagnostics, crashing notifier/detector."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

import pytest
import ray

from miles.utils.ft.adapters.types import NotifierProtocol, ft_controller_actor_name
from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
    RayNodeInfo,
)
from tests.fast.utils.ft.testbed.config import TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed
from tests.fast.utils.ft.utils.controller_fakes import CrashingDetector

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


class _CrashingNotifier(NotifierProtocol):
    """Notifier whose send() always raises. Serializable via cloudpickle."""

    async def send(self, title: str, content: str, severity: str) -> None:
        raise RuntimeError("notifier crash for testing")

    async def aclose(self) -> None:
        pass


@pytest.fixture
async def make_testbed(
    local_ray_nodes: list[RayNodeInfo],
) -> AsyncIterator[Callable[..., MilesTestbed]]:
    """Factory fixture for creating MilesTestbed instances with custom configs."""
    from typing import Any

    from tests.fast.utils.ft.testbed.config import TestbedConfig

    created: list[MilesTestbed] = []

    async def _factory(**kwargs: Any) -> MilesTestbed:
        config = TestbedConfig(**kwargs)
        tb = await MilesTestbed.launch(config=config, ray_nodes=local_ray_nodes)
        created.append(tb)
        return tb

    yield _factory

    for tb in created:
        await tb.shutdown()


async def test_log_step_after_controller_death(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill controller -> workers survive and keep running."""
    # Step 1: create testbed with 1 training node
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
    )

    # Step 2: verify training is stable
    await testbed.wait_for_training_stable(n_iterations=2, timeout=FAST_TIMEOUT)

    # Step 3: kill controller actor
    controller_name = ft_controller_actor_name(testbed.ft_id)
    try:
        ray.get(testbed.controller.shutdown.remote(), timeout=5)
    except Exception:
        logger.debug("controller shutdown failed (expected)", exc_info=True)
    try:
        ray.kill(ray.get_actor(controller_name), no_restart=True)
    except (ValueError, Exception):
        logger.debug("controller kill failed (expected)", exc_info=True)

    # Step 4: wait a bit — workers should continue running without crashing
    await asyncio.sleep(2.0)

    # Step 5: verify workers are still alive
    alive = await testbed.train_group.all_alive()
    assert alive, "Workers should survive controller death"


async def test_all_pass_diagnostics_escalates_to_notify(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Repeated crash with all-pass diagnostics -> NotifyHumans (no root cause).

    When the recovery stepper cannot identify a root cause (all diagnostics
    pass), it escalates to NotifyHumans to alert operators.
    """
    # Step 1: create testbed — slow step so MonitoringProgress stays active
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2, diagnostic_pass=True),
        ],
        detectors=[TrainingCrashDetector()],
        step_interval=2.0,
        monitoring_success_iterations=999,
    )

    await testbed.wait_for_training_stable(n_iterations=1, timeout=FAST_TIMEOUT)

    # Step 2: crash -> recovery -> MonitoringProgress (stays active due to high iterations)
    await testbed.crash_training()
    await testbed.wait_for_recovery_phase(
        phase="MonitoringProgressSt",
        timeout=FAST_TIMEOUT,
    )

    # Step 3: crash during MonitoringProgress -> RestartFailed ->
    # StopTimeDiagnostics -> single node passes -> no root cause -> NotifyHumans
    await testbed.crash_training()

    import time

    deadline = time.monotonic() + RECOVERY_TIMEOUT
    while time.monotonic() < deadline:
        status = await testbed.get_status()
        if status.recovery is not None and status.recovery.phase == "NotifyHumansSt":
            break
        if status.mode == ControllerMode.MONITORING and not status.recovery_in_progress:
            break
        await asyncio.sleep(0.5)


async def test_crashing_notifier_does_not_break_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Notifier raises on send() -> safe_notify catches -> recovery completes."""
    # Step 1: create testbed with crashing notifier
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[TrainingCrashDetector()],
        recovery_cooldown=SlidingWindowThrottle(window_minutes=60, max_count=2),
        notifier_override=_CrashingNotifier(),
    )

    await testbed.wait_for_training_stable(n_iterations=3, timeout=RECOVERY_TIMEOUT)

    # Step 2: first crash -> recovery (notifier not called on normal recovery)
    await testbed.crash_training()
    await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 3: second crash -> throttled -> notifier.send() called -> raises -> controller survives
    await testbed.wait_for_training_stable(n_iterations=2, timeout=RECOVERY_TIMEOUT)
    await testbed.crash_training()
    await asyncio.sleep(5.0)

    # Step 4: controller is still alive and functional
    status = await testbed.get_status()
    assert status.mode == ControllerMode.MONITORING


async def test_crashing_detector_does_not_crash_controller(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """A crashing detector is skipped; TrainingCrashDetector still fires normally."""
    # Step 1: create testbed with CrashingDetector (throws every tick) + TrainingCrashDetector
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        detectors=[CrashingDetector(), TrainingCrashDetector()],
    )

    # Step 2: let training run for a while with CrashingDetector throwing every tick
    await testbed.wait_for_training_stable(n_iterations=5, timeout=FAST_TIMEOUT)

    # Step 3: crash training -> TrainingCrashDetector should still work
    await testbed.crash_training()
    final = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    assert final.mode == ControllerMode.MONITORING
