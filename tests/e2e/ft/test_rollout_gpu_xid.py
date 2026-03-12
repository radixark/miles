"""E2E: GPU XID on rollout node → recovery → node evicted.

Verifies that a hardware fault (GPU XID) on a rollout node triggers recovery
and the node is marked as bad (evicted). Training should remain unaffected.
"""

from __future__ import annotations

import logging

import pytest
import ray
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    ROLLOUT_FOCUSED,
    find_rollout_node,
    get_status,
    wait_for_all_subsystems_detecting,
    wait_for_subsystem_state,
    wait_for_training_stable,
)

from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager

logger = logging.getLogger(__name__)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)
async def test_rollout_gpu_xid_evicts_node(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    handle = ft_controller_handle

    # Step 1: wait for training to stabilize
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    # Step 2: find a rollout node and identify its subsystem
    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    # Step 3: inject GPU XID on the rollout node
    ray.get(rollout_injector.trigger_gpu_xid.remote())
    logger.info("gpu_xid_triggered on node=%s", rollout_node_id)

    # Step 4: wait for rollout subsystem to enter RecoveringSt
    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="RecoveringSt",
        timeout=180.0,
    )
    logger.info("%s entered RecoveringSt", target_subsystem)

    # Step 5: wait for rollout subsystem to return to DetectingAnomalySt
    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="DetectingAnomalySt",
        timeout=420.0,
    )
    logger.info("%s returned to DetectingAnomalySt", target_subsystem)

    # Step 6: assert node was evicted (hardware fault → mark node bad)
    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert rollout_node_id in bad_nodes, (
        f"Rollout node {rollout_node_id} was NOT evicted despite GPU XID. "
        f"bad_nodes={bad_nodes}"
    )

    # Step 7: assert training subsystem unaffected
    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )
