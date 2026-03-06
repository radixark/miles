"""Shared test scenario functions for local_ray and E2E reuse.

Each scenario takes a controller handle, a fault injector, and optional
parameters. The scenario drives the test through a standard sequence:
wait for stable training → inject fault → observe recovery → verify outcome.

Scenarios are agnostic to whether faults are injected by killing real
processes (E2E) or by changing simulated state (local_ray).
"""
from __future__ import annotations

import time

import ray

from miles.utils.ft.models import ControllerMode, ControllerStatus, RecoveryPhase

from tests.fast.utils.ft.helpers.fault_injection import FaultInjectionProtocol


def get_status(handle: ray.actor.ActorHandle, timeout: float = 10) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


async def wait_for_training_stable(
    handle: ray.actor.ActorHandle,
    n_iterations: int = 3,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> None:
    """Poll controller until latest_iteration advances by n_iterations."""
    baseline = (get_status(handle).latest_iteration or 0)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = get_status(handle).latest_iteration or 0
        if current - baseline >= n_iterations:
            return
        await _async_sleep(poll_interval)
    raise TimeoutError(
        f"Training not stable: needed {n_iterations} iterations from baseline={baseline}, "
        f"timeout={timeout}s"
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
        await _async_sleep(poll_interval)
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
        await _async_sleep(poll_interval)

    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.mode == target_mode:
            return status
        await _async_sleep(poll_interval)

    raise TimeoutError(f"Mode transition to {target_mode} did not complete within {timeout}s")


def assert_phase_path_contains(
    status: ControllerStatus,
    required: list[RecoveryPhase],
) -> None:
    """Assert that phase_history contains the required phases as an ordered subsequence."""
    history = status.phase_history
    assert history is not None, "phase_history is None — was recovery never entered?"

    idx = 0
    for phase in history:
        if idx < len(required) and phase == required[idx]:
            idx += 1

    assert idx == len(required), (
        f"Expected phase path to contain {[p.value for p in required]} in order, "
        f"but got history {[p.value for p in history]}"
    )


async def scenario_transient_crash(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    stable_iterations: int = 3,
    stable_timeout: float = 60.0,
    recovery_timeout: float = 120.0,
) -> ControllerStatus:
    """Single crash → auto-recovery → training resumes.

    Returns the final ControllerStatus after recovery completes.
    """
    pre_status = get_status(handle)
    assert pre_status.mode == ControllerMode.MONITORING

    await injector.crash_training()

    status = await wait_for_mode_transition(
        handle=handle,
        target_mode=ControllerMode.MONITORING,
        timeout=recovery_timeout,
    )

    assert status.mode == ControllerMode.MONITORING
    assert status.bad_nodes == []

    return status


async def scenario_no_false_positive(
    handle: ray.actor.ActorHandle,
    *,
    observation_ticks: int = 30,
    poll_interval: float = 0.2,
) -> ControllerStatus:
    """Let the controller run for N ticks with no injected faults.

    Verify it stays in MONITORING mode and never enters recovery.
    """
    initial = get_status(handle)
    target_ticks = initial.tick_count + observation_ticks

    deadline = time.monotonic() + observation_ticks * poll_interval * 3

    while time.monotonic() < deadline:
        status = get_status(handle)
        assert status.mode == ControllerMode.MONITORING, (
            f"Unexpected mode transition to {status.mode} at tick {status.tick_count}"
        )
        if status.tick_count >= target_ticks:
            return status
        await _async_sleep(poll_interval)

    raise TimeoutError(f"Controller did not reach {target_ticks} ticks")


async def scenario_hang_detection(
    handle: ray.actor.ActorHandle,
    injector: FaultInjectionProtocol,
    *,
    hang_timeout: float = 30.0,
) -> ControllerStatus:
    """Inject hang → controller detects stale iteration → ENTER_RECOVERY.

    Returns the ControllerStatus after recovery is triggered.
    """
    pre_status = get_status(handle)
    assert pre_status.mode == ControllerMode.MONITORING

    await injector.inject_hang()

    status = await wait_for_mode(
        handle=handle,
        target_mode=ControllerMode.RECOVERY,
        timeout=hang_timeout,
    )

    assert status.mode == ControllerMode.RECOVERY
    return status


async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
