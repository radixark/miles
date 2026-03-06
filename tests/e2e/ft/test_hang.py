"""E2E: Training hang via SIGSTOP → detection → recovery.

Slowest E2E test due to the hang detection timeout (~5-10 min).
"""

from __future__ import annotations

import logging
import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode, RecoveryPhase
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    assert_phase_path_contains,
    find_training_pid,
    get_status,
    wait_for_mode_transition,
    wait_for_training_stable,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]

_HANG_TIMEOUT_MINUTES = 10
_DETECTION_BUFFER_SECONDS = 60
_MAX_DETECTION_SECONDS = _HANG_TIMEOUT_MINUTES * 60 + _DETECTION_BUFFER_SECONDS


async def test_hang_detection_and_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=5,
        timeout=300.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)
    target_pid = find_training_pid(injector, node_id=target_node)
    t_inject = time.monotonic()
    ray.get(injector.stop_process.remote(pid=target_pid))
    logger.info("SIGSTOP sent to pid=%d on node=%s", target_pid, target_node)

    status = await wait_for_mode_transition(
        handle=ft_controller_handle,
        target_mode=ControllerMode.MONITORING,
        timeout=720.0,
        poll_interval=10.0,
    )

    t_detect = time.monotonic() - t_inject
    logger.info("hang_detected_and_recovered t_detect=%.1fs", t_detect)

    assert status.mode == ControllerMode.MONITORING

    assert t_detect < _MAX_DETECTION_SECONDS, (
        f"Hang detection took {t_detect:.0f}s, expected < {_MAX_DETECTION_SECONDS}s "
        f"(hang_timeout={_HANG_TIMEOUT_MINUTES}min + {_DETECTION_BUFFER_SECONDS}s buffer)"
    )

    final_status = get_status(ft_controller_handle)
    assert_phase_path_contains(final_status, [
        RecoveryPhase.CHECK_ALERTS,
        RecoveryPhase.REATTEMPTING,
        RecoveryPhase.MONITORING,
        RecoveryPhase.DONE,
    ])

    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=10,
        timeout=300.0,
    )
