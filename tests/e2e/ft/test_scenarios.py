"""E2E scenario tests -- thin wrappers calling shared scenario functions.

Previously 9 separate E2E test files each imported scenario functions from
a non-existent scenarios module (ModuleNotFoundError at runtime). This file
consolidates them using the shared scenario functions that work against both
E2E (real cluster) and semi-E2E (MilesTestbed) via FaultTestProtocol.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
import time

import pytest
import ray
from tests.e2e.ft.conftest import (
    E2eFaultInjector,
    E2eFaultTestAdapter,
    FaultInjectorFactory,
    MULTI_CELL,
    ROLLOUT_FOCUSED,
    TRAINING_FOCUSED,
    find_all_rollout_nodes,
    find_rollout_node,
    get_status,
    wait_for_sglang_pid,
    wait_for_training_stable,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    scenario_hang_detection_and_recovery,
    scenario_multi_cell_crash,
    scenario_no_false_positive,
    scenario_repeated_crash,
    scenario_rollout_crash,
    scenario_rollout_gpu_xid,
    scenario_rollout_repeated_crash,
    scenario_transient_crash,
)

from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Training: transient crash
# ------------------------------------------------------------------


async def test_transient_crash_auto_recovery(
    ft_env: E2eFaultTestAdapter,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    """Single kill -> auto-recovery -> training resumes."""
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    t0 = time.monotonic()
    await scenario_transient_crash(
        env=ft_env,
        injector=fault,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
        post_recovery_iterations=5,
        post_recovery_timeout=300.0,
    )
    assert time.monotonic() - t0 < 300.0


# ------------------------------------------------------------------
# Training: Python exception
# ------------------------------------------------------------------


async def test_python_exception_auto_recovery(
    ft_env: E2eFaultTestAdapter,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    """Python exception via flag file -> auto-recovery."""
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    t0 = time.monotonic()
    await scenario_transient_crash(
        env=ft_env,
        injector=fault,
        crash_fn=fault.inject_python_exception,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
        post_recovery_iterations=5,
        post_recovery_timeout=300.0,
    )
    assert time.monotonic() - t0 < 300.0


# ------------------------------------------------------------------
# Training: repeated crash -> DIAGNOSING
# ------------------------------------------------------------------


async def test_repeated_crash_enters_diagnosing(
    ft_env: E2eFaultTestAdapter,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    """Two rapid crashes -> escalation to StopTimeDiagnostics."""
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )
    await scenario_repeated_crash(
        env=ft_env,
        injector=fault,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
    )


# ------------------------------------------------------------------
# Training: hang detection and recovery
# ------------------------------------------------------------------


@pytest.mark.timeout(900)
async def test_hang_detection_and_recovery(
    ft_env: E2eFaultTestAdapter,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    """SIGSTOP -> hang detection -> recovery -> training resumes."""
    await ft_env.wait_for_training_stable(n_iterations=3, timeout=180.0)

    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    await scenario_hang_detection_and_recovery(
        env=ft_env,
        injector=fault,
        hang_timeout=720.0,
        recovery_timeout=720.0,
        max_detection_seconds=660.0,
        post_recovery_iterations=5,
        post_recovery_timeout=300.0,
    )


# ------------------------------------------------------------------
# No false positive
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "ft_controller_handle",
    [TRAINING_FOCUSED, ROLLOUT_FOCUSED],
    indirect=True,
)
async def test_no_false_positive_during_normal_training(
    ft_env: E2eFaultTestAdapter,
) -> None:
    """Controller stays in MONITORING during normal training."""
    status = await scenario_no_false_positive(
        env=ft_env,
        observation_iterations=10,
        timeout=120.0,
        poll_interval=5.0,
    )

    assert status.subsystem_states, "Expected non-empty subsystem_states"
    for name, state in status.subsystem_states.items():
        assert state == "DetectingAnomalySt", (
            f"Subsystem {name} in unexpected state {state}, expected DetectingAnomalySt"
        )


# ------------------------------------------------------------------
# Rollout: single engine crash -> L1 restart
# ------------------------------------------------------------------


@pytest.mark.timeout(600)
@pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)
async def test_rollout_engine_crash_restarts_without_eviction(
    ft_env: E2eFaultTestAdapter,
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    """sglang crash -> L1 restart -> node NOT evicted."""
    handle = ft_controller_handle

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

    status = await scenario_rollout_crash(
        env=ft_env,
        crash_fn=functools.partial(fault.crash_rollout_on_node, rollout_node_id),
        target_subsystem=target_subsystem,
        stable_iterations=3,
        stable_timeout=300.0,
        detection_timeout=180.0,
        recovery_timeout=420.0,
    )

    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert rollout_node_id not in bad_nodes, (
        f"Rollout node {rollout_node_id} was evicted despite software-only crash. "
        f"bad_nodes={bad_nodes}"
    )

    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=120.0)


# ------------------------------------------------------------------
# Rollout: GPU XID -> node evicted
# ------------------------------------------------------------------


@pytest.mark.timeout(600)
@pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)
async def test_rollout_gpu_xid_evicts_node(
    ft_env: E2eFaultTestAdapter,
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    """GPU XID on rollout node -> recovery -> node evicted."""
    handle = ft_controller_handle

    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    async def _inject_xid() -> None:
        ray.get(rollout_injector.trigger_gpu_xid.remote())
        logger.info("gpu_xid_triggered on node=%s", rollout_node_id)

    status = await scenario_rollout_gpu_xid(
        env=ft_env,
        inject_xid_fn=_inject_xid,
        target_subsystem=target_subsystem,
        detection_timeout=180.0,
        recovery_timeout=420.0,
    )

    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert rollout_node_id in bad_nodes, (
        f"Rollout node {rollout_node_id} was NOT evicted despite GPU XID. "
        f"bad_nodes={bad_nodes}"
    )

    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )


# ------------------------------------------------------------------
# Rollout: multi-cell crash -> independent recovery
# ------------------------------------------------------------------


@pytest.mark.timeout(900)
@pytest.mark.parametrize("ft_controller_handle", [MULTI_CELL], indirect=True)
async def test_multi_cell_independent_crash_recovery(
    ft_env: E2eFaultTestAdapter,
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
) -> None:
    """Crash 2 of 3 rollout cells with stagger -> both recover independently."""
    handle = ft_controller_handle

    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    status = get_status(handle)
    rollout_subsystems = sorted(n for n in status.subsystem_states if n.startswith("rollout_"))
    assert len(rollout_subsystems) >= 3, (
        f"Expected at least 3 rollout subsystems, got {len(rollout_subsystems)}: {rollout_subsystems}"
    )

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

    async def _crash_node(injector: ray.actor.ActorHandle, node_id: str) -> None:
        pid = await wait_for_sglang_pid(injector, timeout=30.0)
        ray.get(injector.kill_process.remote(pid=pid, sig=signal.SIGKILL))
        logger.info("killed sglang on node=%s pid=%d", node_id, pid)

    crash_fns = [
        functools.partial(_crash_node, injector_0, node_0_id),
        functools.partial(_crash_node, injector_1, node_1_id),
    ]

    status = await scenario_multi_cell_crash(
        env=ft_env,
        crash_fns=crash_fns,
        all_rollout_subsystems=rollout_subsystems,
        stable_iterations=0,
        stagger_delay=30.0,
        detection_timeout=300.0,
        recovery_timeout=600.0,
    )

    for name in rollout_subsystems:
        assert status.subsystem_states[name] == "DetectingAnomalySt", (
            f"{name} not in DetectingAnomalySt after recovery: {status.subsystem_states[name]}"
        )
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )


# ------------------------------------------------------------------
# Rollout: repeated crash -> escalation -> eviction
# ------------------------------------------------------------------


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
    ft_env: E2eFaultTestAdapter,
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    """Repeatedly kill sglang -> L1 restarts fail -> escalate to node eviction."""
    handle = ft_controller_handle

    await wait_for_training_stable(handle=handle, n_iterations=3, timeout=300.0)

    rollout_node_id, rollout_injector = find_rollout_node(fault_injector)
    status = get_status(handle)
    rollout_subsystems = [n for n in status.subsystem_states if n.startswith("rollout_")]
    assert rollout_subsystems, f"No rollout subsystems found in {status.subsystem_states}"
    target_subsystem = rollout_subsystems[0]
    logger.info("target_rollout_node=%s subsystem=%s", rollout_node_id, target_subsystem)

    stop_event = asyncio.Event()
    kill_task: asyncio.Task[None] | None = None

    async def _start_killing() -> asyncio.Task[None]:
        nonlocal kill_task
        kill_task = asyncio.create_task(
            _keep_killing_sglang(
                injector=rollout_injector,
                node_id=rollout_node_id,
                stop_event=stop_event,
                kill_interval=5.0,
            )
        )
        return kill_task

    async def _stop_killing() -> None:
        stop_event.set()
        if kill_task is not None:
            await kill_task

    async def _check_eviction() -> bool:
        bad_nodes = set(await k8s_node_manager.get_bad_nodes())
        return rollout_node_id in bad_nodes

    status = await scenario_rollout_repeated_crash(
        env=ft_env,
        start_killing_fn=_start_killing,
        stop_killing_fn=_stop_killing,
        check_eviction_fn=_check_eviction,
        target_subsystem=target_subsystem,
        detection_timeout=180.0,
        eviction_timeout=900.0,
        recovery_timeout=420.0,
    )

    status = get_status(handle)
    assert status.subsystem_states.get("training") == "DetectingAnomalySt", (
        f"Training subsystem affected: {status.subsystem_states.get('training')}"
    )
