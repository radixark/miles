"""Protocols for shared E2E / semi-E2E scenario functions.

Previously E2E used ray.actor.ActorHandle + conftest helpers while semi-E2E
used MilesTestbed directly, with completely different APIs. These protocols
define a common interface so scenario functions work with both.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


@runtime_checkable
class FaultTestProtocol(Protocol):
    """Common test-environment operations shared by E2E and semi-E2E."""

    async def get_status(self) -> ControllerStatus: ...

    async def wait_for_training_stable(self, *, n_iterations: int, timeout: float) -> None: ...

    async def wait_for_mode_transition(self, *, target_mode: ControllerMode, timeout: float) -> ControllerStatus: ...

    async def wait_for_subsystem_state(self, *, name: str, state: str, timeout: float) -> ControllerStatus: ...

    async def wait_for_recovery_phase(self, *, phase: str, timeout: float) -> ControllerStatus: ...

    async def wait_for_all_subsystems_detecting(self, *, timeout: float) -> ControllerStatus: ...


@runtime_checkable
class FaultInjectionProtocol(Protocol):
    """Basic fault injection operations available on both E2E and semi-E2E."""

    async def crash_training(self) -> None: ...

    async def inject_hang(self) -> None: ...
