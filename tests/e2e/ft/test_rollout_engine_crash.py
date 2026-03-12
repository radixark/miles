"""E2E: sglang engine crash → Level 1 restart → node NOT evicted.

Verifies that a software-level sglang crash (not hardware) triggers only a
subsystem-level restart (actuator stop+start) without evicting the node.
Training subsystem should remain unaffected throughout.
"""

from __future__ import annotations

import logging

import pytest
import ray
from tests.e2e.ft.conftest import (
    E2eFaultInjector,
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
async def test_rollout_engine_crash_restarts_without_eviction(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    handle = ft_controller_handle

    # Step 1: wait for training to stabilize
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    # Step 2: confirm rollout_0 is in DetectingAnomalySt
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]
    assert status.subsystem_states[target_subsystem] == "DetectingAnomalySt", (
        f"{target_subsystem} not in DetectingAnomalySt: {status.subsystem_states[target_subsystem]}"
    )

    # Step 3: find a rollout node and record its ID
    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=rollout_node_id),
        target_node=rollout_node_id,
        rollout_injectors={rollout_node_id: rollout_injector},
    )

    # Step 4: SIGKILL sglang process on the rollout node
    await fault.crash_rollout_on_node(node_id=rollout_node_id)
    logger.info("sglang_killed on node=%s", rollout_node_id)

    # Step 5: wait for rollout subsystem to enter RecoveringSt (~60-120s, alive_threshold_seconds=60)
    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="RecoveringSt",
        timeout=180.0,
    )
    logger.info("%s entered RecoveringSt", target_subsystem)

    # Step 6: wait for rollout subsystem to return to DetectingAnomalySt (Level 1 restart + sustained_alive ~180s)
    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="DetectingAnomalySt",
        timeout=420.0,
    )
    logger.info("%s returned to DetectingAnomalySt", target_subsystem)

    # Step 7: assert node NOT evicted (software crash should not mark node as bad)
    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert rollout_node_id not in bad_nodes, (
        f"Rollout node {rollout_node_id} was evicted despite software-only crash. "
        f"bad_nodes={bad_nodes}"
    )

    # Step 8: assert training subsystem unaffected
    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )

    # Step 9: assert training still advancing iterations
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=120.0)
