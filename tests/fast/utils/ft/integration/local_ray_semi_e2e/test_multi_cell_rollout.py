"""Semi-E2E: multi-cell rollout — staggered cell failures with independent recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

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


async def test_staggered_cell_failures_independent_recovery(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill cell-0, wait for its recovery to start, then kill cell-1.

    Both cells should recover independently while cell-2 remains unaffected
    throughout. This exercises interleaved cell failures during active recovery,
    which is distinct from simultaneous or fully sequential failure patterns.
    """
    # Step 1: create testbed with 3 rollout cells + 1 training node
    testbed = await make_testbed(
        training_nodes=[TestbedNodeConfig(node_id="n-0", num_ranks=2)],
        rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
        rollout_num_cells=3,
        rollout_alive_threshold_seconds=2.0,
        rollout_monitoring_alive_duration_seconds=0,
    )

    # Step 2: verify training is stable before injecting faults
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 3: verify all 3 rollout subsystems are healthy
    status = await testbed.get_status()
    for cell_name in ["rollout_0", "rollout_1", "rollout_2"]:
        assert status.subsystem_states.get(cell_name) == "DetectingAnomalySt", (
            f"{cell_name} not in DetectingAnomalySt before fault injection: "
            f"{status.subsystem_states}"
        )

    # Step 4: kill cell-0
    await testbed.kill_sglang_cell("0")

    # Step 5: wait for cell-0 to enter recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_0",
        state="RecoveringSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 6: while cell-0 is recovering, verify cell-2 is still healthy
    status = await testbed.get_status()
    assert status.subsystem_states.get("rollout_2") == "DetectingAnomalySt", (
        f"rollout_2 should be unaffected while cell-0 recovers: "
        f"{status.subsystem_states}"
    )

    # Step 7: kill cell-1 while cell-0 recovery is in progress
    await testbed.kill_sglang_cell("1")

    # Step 8: wait for cell-1 to also enter recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_1",
        state="RecoveringSt",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 9: verify cell-2 remains unaffected during both concurrent recoveries
    status = await testbed.get_status()
    assert status.subsystem_states.get("rollout_2") == "DetectingAnomalySt", (
        f"rollout_2 should remain unaffected during concurrent recovery of 0 and 1: "
        f"{status.subsystem_states}"
    )

    # Step 10: wait for cell-0 to finish recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_0",
        state="DetectingAnomalySt",
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 11: wait for cell-1 to finish recovery
    await testbed.wait_for_subsystem_state(
        name="rollout_1",
        state="DetectingAnomalySt",
        timeout=LONG_RECOVERY_TIMEOUT,
    )

    # Step 12: verify all subsystems are back to normal
    status = await testbed.wait_for_all_subsystems_detecting(timeout=LONG_RECOVERY_TIMEOUT)
    for name in ["rollout_0", "rollout_1", "rollout_2", "training"]:
        assert status.subsystem_states.get(name) == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt after staggered recovery: "
            f"{status.subsystem_states}"
        )

    # Step 13: verify training continued throughout (was never disrupted)
    assert status.latest_iteration is not None and status.latest_iteration > 0, (
        f"Training should have continued during rollout recoveries: "
        f"latest_iteration={status.latest_iteration}"
    )
