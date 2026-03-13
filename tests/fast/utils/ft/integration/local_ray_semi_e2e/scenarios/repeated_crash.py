"""Scenario: repeated crash -> reattempt fails -> DIAGNOSING."""

from __future__ import annotations

import asyncio
import time

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios.protocol import (
    FaultInjectionProtocol,
    FaultTestProtocol,
)

from miles.utils.ft.controller.types import ControllerMode


async def scenario_repeated_crash(
    env: FaultTestProtocol,
    injector: FaultInjectionProtocol,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    recovery_timeout: float = 120.0,
) -> None:
    """Two rapid crashes -> recovery attempt fails -> StopTimeDiagnostics.

    First crash triggers recovery. Second crash during MonitoringProgress
    causes the recovery to escalate to StopTimeDiagnostics.
    """
    # Step 1: wait for training to stabilize
    await env.wait_for_training_stable(n_iterations=stable_iterations, timeout=stable_timeout)

    # Step 2: first crash -> enters recovery
    await injector.crash_training()
    await env.wait_for_recovery_phase(phase="MonitoringProgressSt", timeout=recovery_timeout)

    # Step 3: second crash during MonitoringProgress -> escalates
    await injector.crash_training()

    # Step 4: verify escalation to StopTimeDiagnostics
    deadline = time.monotonic() + recovery_timeout
    while time.monotonic() < deadline:
        status = await env.get_status()
        if status.recovery is not None and status.recovery.phase == "StopTimeDiagnosticsSt":
            assert (
                status.mode == ControllerMode.RECOVERY
            ), f"Expected RECOVERY mode during StopTimeDiagnostics, got {status.mode}"
            return
        await asyncio.sleep(0.5)

    raise TimeoutError(f"StopTimeDiagnostics not observed within {recovery_timeout}s")
