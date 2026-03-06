"""Fault injection abstractions for shared test scenarios.

Provides a protocol that both local_ray and E2E tests can implement,
so scenario functions in scenarios.py can inject faults generically.
"""
from __future__ import annotations

from typing import Protocol

import ray

from miles.utils.ft.protocols.platform import JobStatus


class FaultInjectionProtocol(Protocol):
    """Abstract interface for injecting faults into training runs."""

    async def crash_training(self) -> None:
        """Simulate a training crash (job transitions to FAILED)."""
        ...

    async def recover_training(self) -> None:
        """Restore training to RUNNING state (for testing false-positive avoidance)."""
        ...

    async def inject_hang(self) -> None:
        """Simulate a training hang (job reports RUNNING but iteration stalls)."""
        ...


class LocalRayFaultInjector:
    """Fault injector for local_ray tests.

    Controls training state via TrainingStateActor to simulate crashes/hangs
    without actual process kills.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    async def crash_training(self) -> None:
        await self._state.set_status.remote(JobStatus.FAILED.value)

    async def recover_training(self) -> None:
        await self._state.set_status.remote(JobStatus.RUNNING.value)

    async def inject_hang(self) -> None:
        """Training reports RUNNING but iterations stop advancing.

        Since the simulated env never advances iteration metrics anyway,
        this is effectively a no-op — the training is already 'hung' from
        the metric perspective. We just ensure status stays RUNNING.
        """
        await self._state.set_status.remote(JobStatus.RUNNING.value)
