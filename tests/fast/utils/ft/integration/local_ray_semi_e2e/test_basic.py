"""Basic MilesTestbed integration tests: training crash, GPU XID, rollout crash."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, RECOVERY_TIMEOUT, RayNodeInfo
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedConfig, TestbedNodeConfig

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


@pytest.fixture
async def testbed(local_ray_nodes: list[RayNodeInfo]) -> AsyncIterator[MilesTestbed]:
    tb = await MilesTestbed.launch(
        config=TestbedConfig(
            training_nodes=[
                TestbedNodeConfig(node_id="train-0", num_ranks=2),
                TestbedNodeConfig(node_id="train-1", num_ranks=2),
            ],
            rollout_nodes=[TestbedNodeConfig(node_id="rollout-0")],
            spare_nodes=["spare-0"],
            rollout_num_cells=1,
            monitoring_success_iterations=3,
            rollout_alive_threshold_seconds=2.0,
            rollout_monitoring_alive_duration_seconds=0,
        ),
        ray_nodes=local_ray_nodes,
    )
    yield tb
    await tb.shutdown()


# test_training_crash_detected_and_recovered removed: covered by test_scenarios::test_transient_crash


async def test_gpu_xid_evicts_node(testbed: MilesTestbed) -> None:
    """Inject GPU XID on training node → GpuFaultDetector → recovery → eviction."""
    # Step 1: Verify training is stable
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: Inject XID on train-0 (collector baseline overridden)
    await testbed.inject_gpu_xid("train-0")

    # Step 3: GpuFaultDetector detects XID > 0 → enters recovery
    status = await testbed.wait_for_subsystem_state(
        name="training",
        state="Recovering",
        timeout=RECOVERY_TIMEOUT,
    )
    assert status.recovery is not None

    # Step 4: Recovery completes
    await testbed.wait_for_subsystem_state(
        name="training",
        state="DetectingAnomaly",
        timeout=RECOVERY_TIMEOUT,
    )

    # Step 5: Verify node was marked bad
    assert testbed.node_manager.was_ever_marked_bad("train-0")


# test_rollout_crash_detected_and_recovered removed: covered by test_scenarios::test_rollout_crash
