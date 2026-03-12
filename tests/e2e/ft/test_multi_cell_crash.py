"""E2E: Multi-cell independent fault recovery — crash 2 cells, leave 1 alive.

Uses MULTI_CELL topology (3 rollout cells on 3 nodes + 1 training + 0 spare).
Two cells are killed staggered; the third stays alive. Both crashed cells should
recover independently via Level 1 restart (no node eviction needed since it's
a software crash on the same node).
"""

from __future__ import annotations

import asyncio
import logging
import signal

import pytest
import ray
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    MULTI_CELL,
    find_all_rollout_nodes,
    get_status,
    wait_for_all_subsystems_detecting,
    wait_for_subsystem_state,
    wait_for_training_stable,
    wait_for_sglang_pid,
)

logger = logging.getLogger(__name__)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("ft_controller_handle", [MULTI_CELL], indirect=True)
async def test_multi_cell_independent_crash_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
) -> None:
    handle = ft_controller_handle

    # Step 1: wait for training to stabilize
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    # Step 2: confirm rollout_0, rollout_1, rollout_2 all exist
    status = get_status(handle)
    rollout_subsystems = sorted(n for n in status.subsystem_states if n.startswith("rollout_"))
    assert len(rollout_subsystems) >= 3, (
        f"Expected at least 3 rollout subsystems, got {len(rollout_subsystems)}: {rollout_subsystems}"
    )
    for name in rollout_subsystems:
        assert status.subsystem_states[name] == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt: {status.subsystem_states[name]}"
        )

    # Step 3: find all 3 rollout nodes
    rollout_nodes = find_all_rollout_nodes(fault_injector)
    assert len(rollout_nodes) >= 3, (
        f"Expected at least 3 rollout nodes, got {len(rollout_nodes)}"
    )

    node_0_id, injector_0 = rollout_nodes[0]
    node_1_id, injector_1 = rollout_nodes[1]
    node_2_id, injector_2 = rollout_nodes[2]
    logger.info(
        "rollout_nodes: node_0=%s node_1=%s node_2=%s",
        node_0_id, node_1_id, node_2_id,
    )

    # Step 4: SIGKILL sglang on node_0
    pid_0 = await wait_for_sglang_pid(injector_0, timeout=30.0)
    ray.get(injector_0.kill_process.remote(pid=pid_0, sig=signal.SIGKILL))
    logger.info("killed sglang on node_0=%s pid=%d", node_0_id, pid_0)

    # Step 5: wait 30s, then SIGKILL sglang on node_1
    await asyncio.sleep(30.0)
    pid_1 = await wait_for_sglang_pid(injector_1, timeout=30.0)
    ray.get(injector_1.kill_process.remote(pid=pid_1, sig=signal.SIGKILL))
    logger.info("killed sglang on node_1=%s pid=%d", node_1_id, pid_1)

    # Step 6: rollout_2 should remain alive — we verify later

    # Step 7: wait for at least 2 rollout subsystems to enter RecoveringSt
    recovering_count = 0
    for _ in range(60):
        status = get_status(handle)
        recovering_count = sum(
            1 for name in rollout_subsystems
            if status.subsystem_states.get(name) == "RecoveringSt"
        )
        if recovering_count >= 2:
            break
        await asyncio.sleep(5.0)
    assert recovering_count >= 2, (
        f"Expected at least 2 recovering rollout subsystems, got {recovering_count}. "
        f"States: {status.subsystem_states}"
    )
    logger.info("at least 2 rollout subsystems in RecoveringSt")

    # Step 8: wait for all subsystems to return to DetectingAnomalySt (parallel recovery)
    await wait_for_all_subsystems_detecting(handle=handle, timeout=600.0)
    logger.info("all subsystems returned to DetectingAnomalySt")

    # Step 9: assert the surviving rollout subsystem was never affected
    #   (it should have stayed in DetectingAnomalySt throughout — we verify final state)
    status = get_status(handle)
    for name in rollout_subsystems:
        assert status.subsystem_states[name] == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt after recovery: {status.subsystem_states[name]}"
        )

    # Step 10: assert training subsystem unaffected
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )
