"""E2E scenario tests -- thin wrappers calling shared scenario functions.

Previously 9 separate E2E test files each imported scenario functions from
a non-existent scenarios module (ModuleNotFoundError at runtime). This file
consolidates them using the shared scenario functions that work against both
E2E (real cluster) and semi-E2E (MilesTestbed) via FaultTestProtocol.
"""

from __future__ import annotations

import functools
import logging
import signal
import time
from collections.abc import Callable

import pytest
import ray
from tests.e2e.ft.conftest import (
    MULTI_CELL,
    ROLLOUT_FOCUSED,
    TRAINING_FOCUSED,
    E2eFaultInjector,
    E2eFaultTestAdapter,
    FaultInjectorFactory,
    discover_rollout_target,
    find_all_rollout_nodes,
    get_status,
    make_repeated_kill_callbacks,
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
# Training: transient crash / python exception (parametrized)
# ------------------------------------------------------------------


@pytest.mark.parametrize("crash_type", ["kill", "python_exception"])
async def test_transient_crash_auto_recovery(
    ft_env: E2eFaultTestAdapter,
    make_training_fault: Callable[..., E2eFaultInjector],
    target_node: str,
    crash_type: str,
) -> None:
    """Single crash -> auto-recovery -> training resumes."""
    fault = make_training_fault(target_node)
    crash_fn = fault.inject_python_exception if crash_type == "python_exception" else None

    t0 = time.monotonic()
    await scenario_transient_crash(
        env=ft_env,
        injector=fault,
        crash_fn=crash_fn,
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
    make_training_fault: Callable[..., E2eFaultInjector],
    target_node: str,
) -> None:
    """Two rapid crashes -> escalation to StopTimeDiagnostics."""
    fault = make_training_fault(target_node)
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
    make_training_fault: Callable[..., E2eFaultInjector],
    target_node: str,
) -> None:
    """SIGSTOP -> hang detection -> recovery -> training resumes."""
    await ft_env.wait_for_training_stable(n_iterations=3, timeout=180.0)

    fault = make_training_fault(target_node)
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
        assert (
            state == "DetectingAnomalySt"
        ), f"Subsystem {name} in unexpected state {state}, expected DetectingAnomalySt"


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
    target = discover_rollout_target(
        handle=ft_controller_handle,
        factory=fault_injector,
    )

    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target.node_id),
        target_node=target.node_id,
        rollout_injectors={target.node_id: target.injector},
    )

    await scenario_rollout_crash(
        env=ft_env,
        crash_fn=functools.partial(fault.crash_rollout_on_node, target.node_id),
        target_subsystem=target.subsystem_name,
        stable_iterations=3,
        stable_timeout=300.0,
        detection_timeout=180.0,
        recovery_timeout=420.0,
    )

    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert target.node_id not in bad_nodes, (
        f"Rollout node {target.node_id} was evicted despite software-only crash. " f"bad_nodes={bad_nodes}"
    )

    await wait_for_training_stable(handle=ft_controller_handle, n_iterations=3, timeout=120.0)


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
    await wait_for_training_stable(handle=ft_controller_handle, n_iterations=3, timeout=300.0)

    target = discover_rollout_target(
        handle=ft_controller_handle,
        factory=fault_injector,
    )

    async def _inject_xid() -> None:
        ray.get(target.injector.trigger_gpu_xid.remote())
        logger.info("gpu_xid_triggered on node=%s", target.node_id)

    await scenario_rollout_gpu_xid(
        env=ft_env,
        inject_xid_fn=_inject_xid,
        target_subsystem=target.subsystem_name,
        detection_timeout=180.0,
        recovery_timeout=420.0,
    )

    bad_nodes = set(await k8s_node_manager.get_bad_nodes())
    assert target.node_id in bad_nodes, (
        f"Rollout node {target.node_id} was NOT evicted despite GPU XID. " f"bad_nodes={bad_nodes}"
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
    await wait_for_training_stable(handle=ft_controller_handle, n_iterations=3, timeout=300.0)

    status = get_status(ft_controller_handle)
    rollout_subsystems = sorted(n for n in status.subsystem_states if n.startswith("rollout_"))
    assert (
        len(rollout_subsystems) >= 3
    ), f"Expected at least 3 rollout subsystems, got {len(rollout_subsystems)}: {rollout_subsystems}"

    rollout_nodes = find_all_rollout_nodes(fault_injector)
    assert len(rollout_nodes) >= 3, f"Expected at least 3 rollout nodes, got {len(rollout_nodes)}"

    node_0_id, injector_0 = rollout_nodes[0]
    node_1_id, injector_1 = rollout_nodes[1]
    logger.info(
        "rollout_nodes: node_0=%s node_1=%s node_2=%s",
        node_0_id,
        node_1_id,
        rollout_nodes[2][0],
    )

    async def _crash_node(injector: ray.actor.ActorHandle, node_id: str) -> None:
        pid = await wait_for_sglang_pid(injector, timeout=30.0)
        ray.get(injector.kill_process.remote(pid=pid, sig=signal.SIGKILL))
        logger.info("killed sglang on node=%s pid=%d", node_id, pid)

    crash_fns = [
        functools.partial(_crash_node, injector_0, node_0_id),
        functools.partial(_crash_node, injector_1, node_1_id),
    ]

    await scenario_multi_cell_crash(
        env=ft_env,
        crash_fns=crash_fns,
        all_rollout_subsystems=rollout_subsystems,
        stable_iterations=0,
        stagger_delay=30.0,
        detection_timeout=300.0,
        recovery_timeout=600.0,
    )


# ------------------------------------------------------------------
# Rollout: repeated crash -> escalation -> eviction
# ------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("ft_controller_handle", [ROLLOUT_FOCUSED], indirect=True)
async def test_rollout_repeated_crash_escalates_to_eviction(
    ft_env: E2eFaultTestAdapter,
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    k8s_node_manager: K8sNodeManager,
) -> None:
    """Repeatedly kill sglang -> L1 restarts fail -> escalate to node eviction."""
    await wait_for_training_stable(handle=ft_controller_handle, n_iterations=3, timeout=300.0)

    target = discover_rollout_target(
        handle=ft_controller_handle,
        factory=fault_injector,
    )

    async def _check_eviction() -> bool:
        bad_nodes = set(await k8s_node_manager.get_bad_nodes())
        return target.node_id in bad_nodes

    start_killing_fn, stop_killing_fn, _ = make_repeated_kill_callbacks(
        injector=target.injector,
        node_id=target.node_id,
        check_eviction_fn=_check_eviction,
    )

    await scenario_rollout_repeated_crash(
        env=ft_env,
        start_killing_fn=start_killing_fn,
        stop_killing_fn=stop_killing_fn,
        check_eviction_fn=_check_eviction,
        target_subsystem=target.subsystem_name,
        detection_timeout=180.0,
        eviction_timeout=900.0,
        recovery_timeout=420.0,
    )

    status = get_status(ft_controller_handle)
    assert (
        status.subsystem_states.get("training") == "DetectingAnomalySt"
    ), f"Training subsystem affected: {status.subsystem_states.get('training')}"
