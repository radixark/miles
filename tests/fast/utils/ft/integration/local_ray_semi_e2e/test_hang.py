"""Semi-E2E: hang detection — stale iteration, monitoring timeout."""
from __future__ import annotations

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
    scenario_hang_detection,
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
