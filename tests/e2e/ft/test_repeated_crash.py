"""E2E: Repeated crash → reattempt fails → DIAGNOSING."""

from __future__ import annotations

import ray

from tests.e2e.ft.conftest import E2eFaultInjector, FaultInjectorFactory
from tests.fast.utils.ft.helpers.scenarios import scenario_repeated_crash


async def test_repeated_crash_enters_diagnosing(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )
    await scenario_repeated_crash(
        handle=ft_controller_handle,
        injector=fault,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
    )
