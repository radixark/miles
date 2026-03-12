"""E2E: sglang engine crash → Level 1 restart → node NOT evicted.

Verifies that a software-level sglang crash (not hardware) triggers only a
subsystem-level restart (actuator stop+start) without evicting the node.
Training subsystem should remain unaffected throughout.

Uses shared scenario_rollout_crash for the core detection → recovery flow,
then adds E2E-specific assertions (K8s eviction check, training advancement).
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
    wait_for_training_stable,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_rollout_crash

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

    # Step 1: discover rollout node and subsystem
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]

    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=rollout_node_id),
        target_node=rollout_node_id,
        rollout_injectors={rollout_node_id: rollout_injector},
    )

    # Step 2: run shared rollout crash scenario (wait stable → crash → recover → verify training)
    status = await scenario_rollout_crash(
        handle=handle,
        injector=fault,
        target_subsystem=target_subsystem,
        target_node=rollout_node_id,
        stable_iterations=3,
        stable_timeout=300.0,
        detection_timeout=180.0,
        recovery_timeout=420.0,
    )

    # Step 3: E2E-specific: assert node NOT evicted (software crash → no mark_node_bad)
    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert rollout_node_id not in bad_nodes, (
        f"Rollout node {rollout_node_id} was evicted despite software-only crash. "
        f"bad_nodes={bad_nodes}"
    )

    # Step 4: assert training still advancing iterations after recovery
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=120.0)
