"""E2E: sglang repeated crash → escalation → node evicted.

Verifies that when sglang repeatedly crashes (simulating persistent failure),
the system escalates from Level 1 restart to node eviction. A background task
continuously kills sglang to prevent successful recovery.
"""

from __future__ import annotations

import asyncio
import logging
import signal

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


async def _keep_killing_sglang(
    injector: ray.actor.ActorHandle,
    node_id: str,
    stop_event: asyncio.Event,
    kill_interval: float = 5.0,
) -> None:
    """Background coroutine that repeatedly kills sglang on a node until stopped."""
    while not stop_event.is_set():
        try:
            procs = ray.get(injector.find_sglang_processes.remote(), timeout=10)
            for proc in procs:
                try:
                    ray.get(injector.kill_process.remote(pid=proc["pid"], sig=signal.SIGKILL), timeout=5)
                    logger.info("repeated_kill sglang pid=%d node=%s", proc["pid"], node_id)
                except Exception:
                    logger.debug("repeated_kill_failed pid=%d", proc["pid"], exc_info=True)
        except Exception:
            logger.debug("repeated_kill_scan_failed node=%s", node_id, exc_info=True)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=kill_interval)
            break
        except asyncio.TimeoutError:
            pass


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)
async def test_rollout_repeated_crash_escalates_to_eviction(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    handle = ft_controller_handle

    # Step 1: wait for training to stabilize
    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    # Step 2: find a rollout node and record its ID
    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    # Step 3: start background task to continuously kill sglang on the target node
    stop_event = asyncio.Event()
    kill_task = asyncio.create_task(
        _keep_killing_sglang(
            injector=rollout_injector,
            node_id=rollout_node_id,
            stop_event=stop_event,
            kill_interval=5.0,
        )
    )

    try:
        # Step 4: wait for rollout subsystem to enter RecoveringSt (~60-120s)
        await wait_for_subsystem_state(
            handle=handle,
            subsystem_name=target_subsystem,
            target_state="RecoveringSt",
            timeout=180.0,
        )
        logger.info("%s entered RecoveringSt", target_subsystem)

        # Step 5: Level 1 restart attempts fail due to continuous killing.
        #   MonitoringSustainedAlive fails → RestartFailed → StopTimeDiagnostics →
        #   retry → fail again → escalate to node eviction.

        # Step 6: wait for node to be evicted
        deadline = asyncio.get_event_loop().time() + 900.0
        while asyncio.get_event_loop().time() < deadline:
            bad_nodes = set(await k8s_node_manager.get_bad_nodes())
            if rollout_node_id in bad_nodes:
                logger.info("node_evicted node=%s", rollout_node_id)
                break
            await asyncio.sleep(10.0)
        else:
            bad_nodes = set(await k8s_node_manager.get_bad_nodes())
            assert rollout_node_id in bad_nodes, (
                f"Rollout node {rollout_node_id} was NOT evicted after repeated crashes. "
                f"bad_nodes={bad_nodes}"
            )

    finally:
        # Step 7: stop background kill task
        stop_event.set()
        await kill_task

    # Step 8: wait for rollout subsystem to return to DetectingAnomalySt (rebuilt on spare node)
    await wait_for_subsystem_state(
        handle=handle,
        subsystem_name=target_subsystem,
        target_state="DetectingAnomalySt",
        timeout=420.0,
    )
    logger.info("%s returned to DetectingAnomalySt", target_subsystem)

    # Step 9: assert training subsystem unaffected
    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )
