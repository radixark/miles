"""Semi-E2E: peer cascade — killing one node cascades to all ranks via peer-check."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode
from tests.fast.utils.ft.integration.conftest import (
    FAST_TIMEOUT,
    RECOVERY_TIMEOUT,
)
from tests.fast.utils.ft.testbed import MilesTestbed, TestbedNodeConfig

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


# ------------------------------------------------------------------
# 1. test_single_node_kill_cascades_to_all_ranks
# ------------------------------------------------------------------


async def test_single_node_kill_cascades_to_all_ranks(
    make_testbed: Callable[..., MilesTestbed],
) -> None:
    """Kill one node's ranks -> surviving node detects peer death -> exit_actor -> all dead -> recovery.

    Setup: 2 training nodes x 2 ranks = 4 workers with mutual peer-check.
    When node-0's 2 ranks are killed, node-1's 2 ranks detect peer death via
    _all_peers_alive() -> exit_actor(), cascading to a full training crash.
    The controller detects FAILED status and triggers recovery back to MONITORING.
    """
    testbed = await make_testbed(
        training_nodes=[
            TestbedNodeConfig(node_id="n-0", num_ranks=2),
            TestbedNodeConfig(node_id="n-1", num_ranks=2),
        ],
        detectors=[TrainingCrashDetector()],
    )

    # Step 1: verify all 4 ranks are producing iterations
    await testbed.wait_for_training_stable(n_iterations=3, timeout=FAST_TIMEOUT)

    # Step 2: kill only node-0's ranks (2 of 4 workers)
    await testbed.kill_training_on_node("n-0")

    # Step 3: node-1's ranks detect dead peers via peer-check and exit_actor,
    # causing all training workers to die. Controller detects FAILED and
    # enters recovery. Wait for full recovery cycle back to MONITORING.
    status = await testbed.wait_for_mode_transition(
        target_mode=ControllerMode.MONITORING,
        timeout=RECOVERY_TIMEOUT,
    )
    assert status.mode == ControllerMode.MONITORING
