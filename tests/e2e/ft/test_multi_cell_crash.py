"""E2E: Multi-cell independent fault recovery — crash 2 cells, leave 1 alive.

Uses MULTI_CELL topology (3 rollout cells on 3 nodes + 1 training + 0 spare).
Two cells are killed staggered; the third stays alive. Both crashed cells should
recover independently via Level 1 restart (no node eviction needed since it's
a software crash on the same node).

Uses shared scenario_multi_cell_crash for the core stagger+detect+recover flow.
"""

from __future__ import annotations

import logging
import signal

import pytest
import ray
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    MULTI_CELL,
    find_all_rollout_nodes,
    get_status,
    wait_for_sglang_pid,
    wait_for_training_stable,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_multi_cell_crash

logger = logging.getLogger(__name__)


async def _crash_sglang_on_node(
    injector: ray.actor.ActorHandle,
    node_id: str,
) -> None:
    """Find and SIGKILL sglang process on a specific node."""
    pid = await wait_for_sglang_pid(injector, timeout=30.0)
    ray.get(injector.kill_process.remote(pid=pid, sig=signal.SIGKILL))
    logger.info("killed sglang on node=%s pid=%d", node_id, pid)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("ft_controller_handle", [MULTI_CELL], indirect=True)
async def test_multi_cell_independent_crash_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
) -> None:
    handle = ft_controller_handle

    # Step 1: wait for training to stabilize and discover rollout subsystems
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    status = get_status(handle)
    rollout_subsystems = sorted(n for n in status.subsystem_states if n.startswith("rollout_"))
    assert len(rollout_subsystems) >= 3, (
        f"Expected at least 3 rollout subsystems, got {len(rollout_subsystems)}: {rollout_subsystems}"
    )

    # Step 2: find all 3 rollout nodes
    rollout_nodes = find_all_rollout_nodes(fault_injector)
    assert len(rollout_nodes) >= 3, (
        f"Expected at least 3 rollout nodes, got {len(rollout_nodes)}"
    )

    node_0_id, injector_0 = rollout_nodes[0]
    node_1_id, injector_1 = rollout_nodes[1]
    logger.info(
        "rollout_nodes: node_0=%s node_1=%s node_2=%s",
        node_0_id, node_1_id, rollout_nodes[2][0],
    )

    # Step 3: run shared multi-cell crash scenario (crash 2 of 3, stagger, wait recovery)
    crash_fns = [
        lambda: _crash_sglang_on_node(injector_0, node_0_id),
        lambda: _crash_sglang_on_node(injector_1, node_1_id),
    ]

    status = await scenario_multi_cell_crash(
        handle=handle,
        crash_fns=crash_fns,
        all_rollout_subsystems=rollout_subsystems,
        stable_iterations=0,
        stagger_delay=30.0,
        detection_timeout=300.0,
        recovery_timeout=600.0,
    )

    # Step 4: verify all subsystems back to detecting
    for name in rollout_subsystems:
        assert status.subsystem_states[name] == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt after recovery: {status.subsystem_states[name]}"
        )
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )
