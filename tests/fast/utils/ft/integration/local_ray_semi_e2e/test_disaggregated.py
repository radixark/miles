"""Semi-E2E: disaggregated recovery — concurrent independent subsystem recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest
from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_simultaneous_training_and_rollout_crash
# ------------------------------------------------------------------


async def test_simultaneous_training_and_rollout_crash(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Crash training AND kill sglang cell simultaneously -> both subsystems
    independently enter recovery -> both complete recovery -> back to normal."""

    # Step 1: create testbed with training + rollout
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=1,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
        detectors=[TrainingCrashDetector()],
    )

    # Step 2: verify training is stable and all subsystems are healthy
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)
    status = await testbed.get_status()
    assert status.subsystem_states.get("training") == "DetectingAnomalySt"
    assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
    rollout_actor_id_before = await testbed.get_sglang_cell_actor_id("default")
    assert rollout_actor_id_before is not None

    # Step 3: simultaneously crash training AND kill sglang cell
    await asyncio.gather(
        testbed.crash_training(),
        testbed.kill_sglang_cell("default"),
    )

    # Step 4: wait for training subsystem to enter recovery (main SM goes to RECOVERY mode)
    await testbed.wait_for_mode(
        mode=ControllerMode.RECOVERY,
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 5: wait for rollout engine to be recreated and both subsystems to recover
    deadline = asyncio.get_running_loop().time() + LONG_RECOVERY_TIMEOUT
    status = await testbed.get_status()
    rollout_actor_id_after = rollout_actor_id_before
    while asyncio.get_running_loop().time() < deadline:
        status = await testbed.wait_for_all_subsystems_detecting(
            timeout=min(5.0, deadline - asyncio.get_running_loop().time()),
        )
        rollout_actor_id_after = await testbed.get_sglang_cell_actor_id("default")
        if rollout_actor_id_after is not None and rollout_actor_id_after != rollout_actor_id_before:
            break
    else:
        raise TimeoutError("Rollout engine was not recreated after simultaneous training and rollout crash")

    assert rollout_actor_id_after != rollout_actor_id_before
    assert (
        status.subsystem_states.get("training") == "DetectingAnomalySt"
    ), f"Training not recovered: {status.subsystem_states}"
    assert (
        status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
    ), f"Rollout not recovered: {status.subsystem_states}"

    # Step 6: verify controller is back to MONITORING (no active recovery)
    assert status.mode == ControllerMode.MONITORING
    assert status.recovery_in_progress is False
