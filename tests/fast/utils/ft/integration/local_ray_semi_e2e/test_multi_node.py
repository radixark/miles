"""Semi-E2E: multi-node — registration, parallel ranks, grace period."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _SLOW_STEP,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestMultiNode:
    async def test_multi_rank_registration_and_targeted_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """4 ranks across 2 nodes register correctly; crash during MONITORING → DIAGNOSING."""
        env = make_e2e_env(
            ft_id="e2emn",
            nodes=[
                NodeSpec(node_id="e2emn-node-0", num_ranks=2),
                NodeSpec(node_id="e2emn-node-1", num_ranks=2),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            recovery_cooldown=SlidingWindowThrottle(window_minutes=1.0, max_count=2),
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        status = get_status(env.controller)
        assert status.mode == ControllerMode.MONITORING

        # Step 1: crash → recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING
        await env.injector.crash_training()

        # Poll for DIAGNOSING in phase_history during the active recovery.
        # After recovery completes, the FAILED status can auto-trigger a second
        # recovery that overwrites _last_phase_history, so we observe DIAGNOSING
        # while the recovery is still in progress.
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "StopTimeDiagnostics" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("DIAGNOSING not observed in phase_history within 90s")

        assert_phase_path_contains(status, ["StopTimeDiagnostics"])


class TestConcurrentRegistration:
    async def test_parallel_rank_registration(
        self, e2e_multi_node_env: E2EEnv,
    ) -> None:
        """4 workers register in parallel → all ranks visible in controller status."""
        env = e2e_multi_node_env

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        status = get_status(env.controller)
        assert status.latest_iteration is not None
        assert status.latest_iteration > 0


class TestRegistrationGrace:
    async def test_detectors_suppressed_during_grace_period(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """With grace_ticks=10, detectors don't fire during the grace period."""
        env = make_e2e_env(
            ft_id="e2egrce",
            nodes=[NodeSpec(node_id="e2egrce-node-0")],
            detectors=[TrainingCrashDetector()],
            registration_grace_ticks=10,
        )

        # Step 1: crash during grace period → should NOT trigger recovery
        await env.injector.crash_training()
        await asyncio.sleep(0.5)
        status = get_status(env.controller)
        if status.tick_count <= 10:
            assert status.mode == ControllerMode.MONITORING, (
                f"Recovery triggered during grace period at tick {status.tick_count}"
            )

        # Step 2: wait for grace period to end
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.tick_count > 10:
                break
            await asyncio.sleep(0.2)

        # Step 3: crash again after grace period → should trigger recovery
        await env.injector.recover_training()
        await asyncio.sleep(1.0)
        await env.injector.crash_training()
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY
