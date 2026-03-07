"""Semi-E2E: hang detection — stale iteration, monitoring timeout, full recovery cycle."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _SLOW_STEP,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    scenario_hang_detection,
    scenario_hang_detection_and_recovery,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestHangDetection:
    async def test_stale_iteration_triggers_recovery(
        self, e2e_hang_env: E2EEnv,
    ) -> None:
        """Worker iteration stalls → FastHangDetector → ENTER_RECOVERY."""
        status = await scenario_hang_detection(
            handle=e2e_hang_env.controller,
            injector=e2e_hang_env.injector,
            hang_timeout=20.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestMonitoringTimeout:
    async def test_crash_during_hung_monitoring_escalates_to_diagnosing(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Worker hung during recovery MONITORING → crash again → DIAGNOSING.

        Uses slow step_interval so MONITORING lasts long enough to inject crash.
        """
        env = make_e2e_env(
            ft_id="e2emto",
            nodes=[NodeSpec(node_id="e2emto-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])


class TestHangFullRecovery:
    async def test_hang_detection_full_recovery_and_resume(
        self, e2e_hang_env: E2EEnv,
    ) -> None:
        """Hang → detect → full recovery → training resumes with new run_id."""
        status = await scenario_hang_detection_and_recovery(
            handle=e2e_hang_env.controller,
            injector=e2e_hang_env.injector,
            hang_timeout=20.0,
            recovery_timeout=120.0,
            post_recovery_iterations=3,
        )
        assert status.mode == ControllerMode.MONITORING


class TestMonitoringProgressTimeout:
    async def test_monitoring_progress_timeout_without_new_fault(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """MonitoringProgress times out due to no iteration progress → StopTimeDiagnostics."""
        env = make_e2e_env(
            ft_id="e2empt",
            nodes=[NodeSpec(node_id="e2empt-node-0")],
            detectors=[TrainingCrashDetector()],
            step_interval=999.0,
            monitoring_timeout_seconds=5,
            monitoring_success_iterations=100,
        )

        # Step 1: crash → recovery → reaches MonitoringProgress but never progresses
        await env.injector.crash_training()

        # Step 2: wait for StopTimeDiagnostics to appear due to monitoring timeout
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "StopTimeDiagnostics" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                "MonitoringProgress did not timeout to StopTimeDiagnostics within 60s"
            )

        assert_phase_path_contains(status, ["MonitoringProgress", "StopTimeDiagnostics"])
