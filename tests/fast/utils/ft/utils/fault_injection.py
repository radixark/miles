"""Fault injection abstractions for shared test scenarios.

Provides a protocol that both local_ray and E2E tests can implement,
so scenario functions in scenarios.py can inject faults generically.
"""

from __future__ import annotations

from typing import Protocol

import ray

from miles.utils.ft.adapters.types import JobStatus


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

    async def inject_nan_loss(self) -> None:
        """Inject loss=NaN into training worker's log_step metrics."""
        ...

    async def clear_nan_loss(self) -> None:
        """Remove injected NaN loss, returning to normal metrics."""
        ...

    async def inject_python_exception(self) -> None:
        """Trigger a Python exception inside the training process."""
        ...

    async def crash_rollout_on_node(self, node_id: str) -> None:
        """SIGKILL the sglang process on a specific rollout node."""
        ...


class LocalRayFaultInjector:
    """Fault injector for local_ray tests.

    Controls training state via TrainingStateActor to simulate crashes/hangs
    without actual process kills. For rollout faults, delegates to
    FakeRolloutManagerActor to toggle rollout_cell_alive metrics.
    """

    def __init__(
        self,
        state_actor: ray.actor.ActorHandle,
        *,
        rollout_manager: ray.actor.ActorHandle | None = None,
        rollout_node_to_cell: dict[str, str] | None = None,
    ) -> None:
        self._state = state_actor
        self._rollout_manager = rollout_manager
        self._rollout_node_to_cell = rollout_node_to_cell or {}

    async def crash_training(self) -> None:
        await self._state.set_status.remote(JobStatus.FAILED.value)

    async def recover_training(self) -> None:
        await self._state.set_status.remote(JobStatus.RUNNING.value)

    async def inject_hang(self) -> None:
        """Training reports RUNNING but iterations stop advancing.

        Sets the hung flag on TrainingStateActor so that
        TrainingWorkerActor (if present) stops advancing iterations.
        """
        await self._state.set_status.remote(JobStatus.RUNNING.value)
        await self._state.set_hung.remote(True)

    async def inject_nan_loss(self) -> None:
        await self._state.set_custom_log_metrics.remote({"loss": float("nan")})

    async def clear_nan_loss(self) -> None:
        await self._state.set_custom_log_metrics.remote({})

    async def inject_custom_metrics(self, metrics: dict[str, float]) -> None:
        """Inject arbitrary key-value metrics into the training worker's log_step."""
        await self._state.set_custom_log_metrics.remote(metrics)

    async def inject_python_exception(self) -> None:
        await self._state.set_status.remote(JobStatus.FAILED.value)

    async def crash_rollout_on_node(self, node_id: str) -> None:
        cell_id = self._rollout_node_to_cell.get(node_id)
        if cell_id is None:
            raise KeyError(f"No rollout cell mapped to node {node_id}")
        if self._rollout_manager is None:
            raise RuntimeError("No rollout manager; build env with rollout_num_cells > 0")
        await self._rollout_manager.set_cell_alive.remote(cell_id, False)
