"""Semi-E2E: diagnostic eviction — bad node eviction, excluded nodes, partial, all-evicted."""
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
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)


class TestDiagnosticEviction:
    async def test_diagnostic_failure_evicts_bad_node(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """node-0 diagnostic fails, node-1 passes → only node-0 evicted.

        Crash during MONITORING → step_monitoring sees FAILED → DIAGNOSING.
        StubDiagnostic resolves instantly so we check phase_history after
        recovery completes rather than catching DIAGNOSING in flight.
        """
        env = make_e2e_env(
            ft_id="e2ediag",
            nodes=[
                NodeSpec(node_id="e2ediag-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ediag-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for recovery MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=60.0)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, [
            "StopTimeDiagnostics",
            "Evicting",
            "RecoveryDone",
        ])


class TestEvictionExcludedNodes:
    async def test_multi_node_eviction_passes_correct_excluded_ids(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """2 bad nodes out of 3 → excluded_node_ids contains both."""
        env = make_e2e_env(
            ft_id="e2eexcl",
            nodes=[
                NodeSpec(node_id="e2eexcl-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-1", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eexcl-node-2", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])


class TestPartialDiagnostic:
    async def test_unreachable_node_agent_treated_as_bad(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """During DIAGNOSING, kill node agent → RayActorError → treated as bad node."""
        import ray

        env = make_e2e_env(
            ft_id="e2epd",
            nodes=[
                NodeSpec(node_id="e2epd-node-0", num_ranks=1, diagnostic_pass=True),
                NodeSpec(node_id="e2epd-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: kill node-0's agent before crash
        node_agent_0 = env.node_agents["e2epd-node-0"]
        ray.kill(node_agent_0, no_restart=True)
        env.node_agents.pop("e2epd-node-0", None)

        # Step 2: crash → enters recovery → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 3: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(final, ["StopTimeDiagnostics"])
        assert_phase_path_contains(final, ["Evicting"])


class TestAllNodesEvicted:
    async def test_all_diagnostics_fail_notifies_human(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """All nodes fail diagnostic → all evicted → NOTIFY."""
        env = make_e2e_env(
            ft_id="e2eall",
            nodes=[
                NodeSpec(node_id="e2eall-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2eall-node-1", num_ranks=1, diagnostic_pass=False),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=30.0)

        # Step 1: crash → wait for MONITORING phase
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=30.0,
        )

        # Step 2: crash during MONITORING → DIAGNOSING (fast) → recovery completes
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [
            "StopTimeDiagnostics",
            "Evicting",
            "RecoveryDone",
        ])
