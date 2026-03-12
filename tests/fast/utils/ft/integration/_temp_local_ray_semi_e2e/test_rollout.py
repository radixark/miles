"""Semi-E2E: rollout crash recovery — single cell, multi-cell independent failures."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    scenario_multi_cell_crash,
    scenario_rollout_crash,
    wait_for_all_subsystems_detecting,
    wait_for_subsystem_state,
    wait_for_training_stable,
)

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector


class TestRolloutCrash:
    async def test_single_cell_crash_and_recovery(self, e2e_rollout_env: E2EEnv) -> None:
        """Rollout cell crash → RolloutCrashDetector fires → L1 restart → DetectingAnomaly."""
        env = e2e_rollout_env
        target_node = f"{env.ft_id}-rollout-default"

        status = await scenario_rollout_crash(
            handle=env.controller,
            injector=env.injector,
            target_subsystem="rollout_default",
            target_node=target_node,
            stable_iterations=3,
            stable_timeout=FAST_TIMEOUT,
            detection_timeout=RECOVERY_TIMEOUT,
            recovery_timeout=LONG_RECOVERY_TIMEOUT,
        )

        assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
        assert status.subsystem_states.get("training") == "DetectingAnomalySt"


class TestRolloutTrainingIsolation:
    async def test_rollout_crash_does_not_affect_training(
        self,
        e2e_rollout_env: E2EEnv,
    ) -> None:
        """Training continues advancing iterations while rollout recovers."""
        env = e2e_rollout_env
        target_node = f"{env.ft_id}-rollout-default"

        # Step 1: wait for training to stabilize
        await wait_for_training_stable(
            env.controller, n_iterations=3, timeout=FAST_TIMEOUT,
        )

        # Step 2: record baseline iteration
        baseline = get_status(env.controller).latest_iteration or 0

        # Step 3: crash rollout
        await env.injector.crash_rollout_on_node(target_node)

        # Step 4: wait for rollout to enter recovery
        await wait_for_subsystem_state(
            env.controller, "rollout_default", "RecoveringSt",
            timeout=RECOVERY_TIMEOUT,
        )

        # Step 5: verify training kept advancing during rollout recovery
        await asyncio.sleep(2.0)
        current = get_status(env.controller).latest_iteration or 0
        assert current > baseline, (
            f"Training stalled during rollout recovery: baseline={baseline} current={current}"
        )

        # Step 6: wait for rollout recovery to complete
        await wait_for_subsystem_state(
            env.controller, "rollout_default", "DetectingAnomalySt",
            timeout=LONG_RECOVERY_TIMEOUT,
        )


class TestMultiCellCrash:
    async def test_two_of_three_cells_crash_independently_recover(
        self,
        e2e_multi_cell_rollout_env: E2EEnv,
    ) -> None:
        """Crash 2 of 3 rollout cells → both recover independently, third unaffected."""
        env = e2e_multi_cell_rollout_env
        all_subsystems = [f"rollout_{cid}" for cid in env.rollout_cell_ids]

        # Step 1: crash cells 0 and 1, leave cell 2 alive
        crash_fns = [
            lambda: env.injector.crash_rollout_on_node(f"{env.ft_id}-rollout-0"),
            lambda: env.injector.crash_rollout_on_node(f"{env.ft_id}-rollout-1"),
        ]

        status = await scenario_multi_cell_crash(
            handle=env.controller,
            crash_fns=crash_fns,
            all_rollout_subsystems=all_subsystems,
            stable_iterations=3,
            stable_timeout=FAST_TIMEOUT,
            stagger_delay=2.0,
            detection_timeout=RECOVERY_TIMEOUT,
            recovery_timeout=LONG_RECOVERY_TIMEOUT,
        )

        # Step 2: verify all subsystems back to detecting
        for name in all_subsystems:
            assert status.subsystem_states.get(name) == "DetectingAnomalySt", (
                f"{name} not in DetectingAnomalySt: {status.subsystem_states}"
            )
        assert status.subsystem_states.get("training") == "DetectingAnomalySt"


class TestRolloutNoFalsePositive:
    async def test_no_recovery_when_all_cells_healthy(
        self,
        e2e_rollout_env: E2EEnv,
    ) -> None:
        """Healthy rollout cell does not trigger recovery."""
        env = e2e_rollout_env

        await wait_for_training_stable(
            env.controller, n_iterations=5, timeout=FAST_TIMEOUT,
        )

        # Step 1: wait additional ticks with no faults
        await asyncio.sleep(5.0)

        # Step 2: verify all subsystems remain in DetectingAnomaly
        status = get_status(env.controller)
        assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt", (
            f"Unexpected rollout state: {status.subsystem_states}"
        )
        assert status.subsystem_states.get("training") == "DetectingAnomalySt"


class TestRolloutSequentialRecovery:
    async def test_two_sequential_rollout_crashes_both_recover(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → recover → crash again → recover again. No state drift."""
        env = make_e2e_env(
            ft_id="e2erosq",
            nodes=[NodeSpec(node_id="e2erosq-node-0")],
            detectors=[TrainingCrashDetector()],
            scrape_interval_seconds=0.5,
            rollout_num_cells=1,
            rollout_alive_threshold_seconds=2.0,
            rollout_monitoring_alive_duration_seconds=0,
        )
        target_node = f"{env.ft_id}-rollout-default"

        for cycle in range(2):
            await wait_for_training_stable(
                env.controller, n_iterations=3, timeout=FAST_TIMEOUT,
            )

            # Step: crash and wait for recovery
            await env.injector.crash_rollout_on_node(target_node)
            await wait_for_subsystem_state(
                env.controller, "rollout_default", "RecoveringSt",
                timeout=RECOVERY_TIMEOUT,
            )
            await wait_for_subsystem_state(
                env.controller, "rollout_default", "DetectingAnomalySt",
                timeout=LONG_RECOVERY_TIMEOUT,
            )

        status = get_status(env.controller)
        assert status.subsystem_states.get("rollout_default") == "DetectingAnomalySt"
        assert status.subsystem_states.get("training") == "DetectingAnomalySt"
