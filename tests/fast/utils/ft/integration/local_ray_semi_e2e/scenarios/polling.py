"""Polling utilities for E2E and local_ray scenario tests.

Provides async helpers to wait for controller state transitions,
training progress, and recovery phases.
"""

from __future__ import annotations

import asyncio
import time

import ray

from miles.utils.ft.controller.types import ControllerMode, ControllerStatus


def get_status(handle: ray.actor.ActorHandle, timeout: float = 10) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


async def wait_for_training_stable(
    handle: ray.actor.ActorHandle,
    n_iterations: int = 3,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> None:
    """Poll controller until latest_iteration advances by n_iterations."""
    baseline = get_status(handle).latest_iteration or 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = get_status(handle).latest_iteration or 0
        if current - baseline >= n_iterations:
            return
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Training not stable: needed {n_iterations} iterations from baseline={baseline}, " f"timeout={timeout}s"
    )


async def wait_for_mode(
    handle: ray.actor.ActorHandle,
    target_mode: ControllerMode,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> ControllerStatus:
    """Poll until controller mode equals target_mode."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode == target_mode:
            return status
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Mode did not reach {target_mode} within {timeout}s")


async def wait_for_mode_transition(
    handle: ray.actor.ActorHandle,
    target_mode: ControllerMode,
    timeout: float = 120.0,
    poll_interval: float = 0.5,
) -> ControllerStatus:
    """Wait for mode to leave target_mode, then return to it.

    Avoids the race where the mode is already at the target before
    the fault has been detected.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode != target_mode:
            break
        await asyncio.sleep(poll_interval)

    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode == target_mode:
            return status
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Mode transition to {target_mode} did not complete within {timeout}s")


async def wait_for_recovery_phase(
    handle: ray.actor.ActorHandle,
    phase: str,
    timeout: float = 120.0,
    poll_interval: float = 0.5,
) -> ControllerStatus:
    """Poll until recovery.phase matches the target phase (state class name)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.recovery is not None and status.recovery.phase == phase:
            return status
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Recovery phase did not reach {phase} within {timeout}s")


async def wait_for_recovery_complete(
    handle: ray.actor.ActorHandle,
    timeout: float = 120.0,
    poll_interval: float = 0.5,
) -> ControllerStatus:
    """Poll until mode returns to MONITORING (recovery finished)."""
    return await wait_for_mode(
        handle=handle,
        target_mode=ControllerMode.MONITORING,
        timeout=timeout,
        poll_interval=poll_interval,
    )
