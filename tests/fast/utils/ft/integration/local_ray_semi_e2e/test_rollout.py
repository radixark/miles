"""Semi-E2E: rollout crash recovery — single cell, multi-cell, isolation, sequential."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT, RayNodeInfo
from tests.fast.utils.ft.testbed.config import TestbedConfig, TestbedNodeConfig
from tests.fast.utils.ft.testbed.train import MilesTestbed

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


@pytest.fixture
async def make_testbed(
    local_ray_nodes: list[RayNodeInfo],
) -> AsyncIterator[Callable[..., MilesTestbed]]:
    """Factory fixture for creating MilesTestbed instances with custom configs."""
    created: list[MilesTestbed] = []

    async def _factory(**kwargs: Any) -> MilesTestbed:
        config = TestbedConfig(**kwargs)
        tb = await MilesTestbed.launch(config=config, ray_nodes=local_ray_nodes)
        created.append(tb)
        return tb

    yield _factory

    for tb in created:
        await tb.shutdown()


# test_single_cell_crash_and_recovery removed: covered by test_scenarios::test_rollout_crash


async def test_rollout_crash_does_not_affect_training(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Training continues advancing iterations while rollout crash notifies humans."""
    # Step 1: create testbed with training + rollout
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=1,
        scrape_interval_seconds=0.3,
        rollout_alive_threshold_seconds=1.5,
        rollout_monitoring_alive_duration_seconds=0,
    )

    # Step 2: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 3: record baseline iteration
    baseline_status = await testbed.get_status()
    baseline = baseline_status.latest_iteration or 0

    # Step 4: kill rollout cell
    await testbed.kill_sglang_cell("default")

    # Step 5: wait for rollout to enter recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_default",
        state="Recovering",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 6: verify training kept advancing during rollout recovery
    await asyncio.sleep(2.0)
    current_status = await testbed.get_status()
    current = current_status.latest_iteration or 0
    assert current > baseline, f"Training stalled during rollout recovery: baseline={baseline} current={current}"

    # Step 7: wait for rollout recovery to complete
    await testbed.wait_for_subsystem_state(
        name="rollout_default",
        state="DetectingAnomaly",
        timeout=LONG_RECOVERY_TIMEOUT,
    )


# test_two_of_three_cells_crash_independently_recover removed: covered by test_scenarios::test_multi_cell_crash


async def test_no_recovery_when_all_cells_healthy(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Healthy rollout cells do not trigger recovery."""
    # Step 1: create testbed with rollout
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=1,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
    )

    # Step 2: wait for stable state
    await testbed.wait_for_training_stable(n_iterations=5, timeout=FAST_TIMEOUT)

    # Step 3: wait additional time with no faults
    await asyncio.sleep(5.0)

    # Step 4: verify all subsystems remain in DetectingAnomaly
    status = await testbed.get_status()
    assert (
        status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
    ), f"Unexpected rollout state: {status.subsystem_states}"
    assert status.subsystem_states.get("training") == "DetectingAnomalySt"


async def test_two_sequential_rollout_crashes_both_recover(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Repeated rollout crashes notify without perturbing training state."""
    # Step 1: create testbed with 1 rollout cell
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        detectors=[TrainingCrashDetector()],
        scrape_interval_seconds=0.3,
        rollout_num_cells=1,
        rollout_alive_threshold_seconds=1.5,
        rollout_monitoring_alive_duration_seconds=0,
        recovery_cooldown=SlidingWindowThrottle(window_minutes=0.1, max_count=100),
    )

    for _cycle in range(2):
        # Step 2: wait for stable state
        await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

        # Step 3: kill rollout cell and wait for recovery
        await testbed.kill_sglang_cell("default")
        await testbed.wait_for_subsystem_state(
            name="rollout_default",
            state="Recovering",
            timeout=RECOVERY_TIMEOUT,
        )
        await testbed.wait_for_subsystem_state(
            name="rollout_default",
            state="DetectingAnomaly",
            timeout=LONG_RECOVERY_TIMEOUT,
        )

    # Step 4: verify final state
    status = await testbed.get_status()
    assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
    assert status.subsystem_states.get("training") == "DetectingAnomalySt"
    assert len(testbed.notifications) >= 2, "Expected repeated rollout crashes to emit notifications"


async def test_rollout_kill_triggers_recovery_via_health_checker(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill rollout cell → health checker detects dead engine → recovery → restart.

    Verifies the full rollout crash recovery path: RolloutCrashDetector sees
    persistent rollout_cell_alive=0 → ENTER_RECOVERY → RecoveringSt → restart →
    DetectingAnomalySt. The health checker timeout fix (sync→async) enables
    asyncio.wait_for to properly cancel dead-engine probes.
    """
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=1,
        scrape_interval_seconds=0.3,
        rollout_alive_threshold_seconds=1.5,
        rollout_monitoring_alive_duration_seconds=0,
    )

    # Step 1: wait for training to stabilize
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: kill rollout cell — health checker will detect dead engine
    await testbed.kill_sglang_cell("default")

    # Step 3: wait for rollout subsystem to enter recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_default",
        state="Recovering",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 4: wait for recovery to complete
    status = await testbed.wait_for_subsystem_state(
        name="rollout_default",
        state="DetectingAnomaly",
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 5: verify rollout recovered and training was unaffected
    assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
    assert status.subsystem_states.get("training") == "DetectingAnomalySt"
    assert status.mode == ControllerMode.MONITORING
