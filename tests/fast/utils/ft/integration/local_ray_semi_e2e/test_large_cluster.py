"""Semi-E2E: large-scale mixed topology — 3 training nodes + 2 rollout cells."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    LONG_RECOVERY_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


async def test_three_training_two_rollout_independent(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """3 training nodes (2 ranks each) + 2 rollout cells: crash training -> recovery completes, rollout unaffected."""
    # Step 1: create testbed with 3 training nodes + 2 rollout cells (5 Ray nodes total)
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
            TestbedNodeConfig(node_id="n-2", num_ranks=2),
        ],
        rollout_nodes=[
            TestbedNodeConfig(node_id="rollout-0"),
            TestbedNodeConfig(node_id="rollout-1"),
        ],
        rollout_num_cells=2,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
        detectors=[TrainingCrashDetector()],
    )

    # Step 2: verify training is stable across all 3 nodes (6 ranks total)
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 3: verify all subsystems start in DetectingAnomaly
    status = await testbed.get_status()
    for name in ["training", "rollout_0", "rollout_1"]:
        assert status.subsystem_states.get(name) == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt before crash: {status.subsystem_states}"
        )

    # Step 4: crash all training workers
    await testbed.crash_training()

    # Step 5: wait for training subsystem to enter recovery
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 6: verify rollout subsystems remain in DetectingAnomaly during training recovery
    await asyncio.sleep(2.0)
    mid_recovery_status = await testbed.get_status()
    assert mid_recovery_status.subsystem_states.get("rollout_0") == "DetectingAnomalySt", (
        f"rollout_0 was affected by training crash: {mid_recovery_status.subsystem_states}"
    )
    assert mid_recovery_status.subsystem_states.get("rollout_1") == "DetectingAnomalySt", (
        f"rollout_1 was affected by training crash: {mid_recovery_status.subsystem_states}"
    )

    # Step 7: wait for recovery to complete -> back to MONITORING
    final_status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=LONG_RECOVERY_TIMEOUT,
    )
    assert final_status.mode == ControllerMode.MONITORING

    # Step 8: verify training resumes with new workers
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 9: verify all subsystems back to DetectingAnomaly
    status = await testbed.wait_for_all_subsystems_detecting(timeout=FAST_TIMEOUT)
    for name in ["training", "rollout_0", "rollout_1"]:
        assert status.subsystem_states.get(name) == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt after recovery: {status.subsystem_states}"
        )
