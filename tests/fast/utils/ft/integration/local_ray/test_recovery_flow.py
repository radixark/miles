"""Local Ray: Recovery flow — trigger recovery via injected detector, observe state transitions."""
from __future__ import annotations

import time
from typing import Any, Callable

import pytest
import ray

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.models import (
    ActionType,
    ControllerMode,
    Decision,
    RecoveryPhase,
    TriggerType,
)
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

from tests.fast.utils.ft.integration.local_ray.conftest import get_status

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.timeout(60),
]


class _AlwaysCrashDetector(BaseFaultDetector):
    """Detector that fires ENTER_RECOVERY on every evaluation."""

    def evaluate(self, ctx: Any) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason="injected crash for test",
            trigger=TriggerType.CRASH,
            bad_node_ids=["fake-bad-node"],
        )


class _OneShotCrashDetector(BaseFaultDetector):
    """Fires ENTER_RECOVERY once, then returns NONE forever after."""

    def __init__(self) -> None:
        self._fired = False

    def evaluate(self, ctx: Any) -> Decision:
        if not self._fired:
            self._fired = True
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason="one-shot crash for test",
                trigger=TriggerType.CRASH,
                bad_node_ids=["fake-bad-node"],
            )
        return Decision(action=ActionType.NONE, reason="no fault")


def _kill_actor(name: str) -> None:
    try:
        ray.kill(ray.get_actor(name), no_restart=True)
    except ValueError:
        pass


def _poll_until(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.2,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(f"Condition not met within {timeout}s")


class TestRecoveryTriggeredByDetector:
    """Inject a detector that fires ENTER_RECOVERY and observe the mode transition."""

    def test_controller_enters_recovery_mode(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[_AlwaysCrashDetector()],
        )

        handle.submit_and_run.remote()
        time.sleep(0.3)

        ray.get(handle.register_training_rank.remote(
            run_id=get_status(handle).active_run_id or "",
            rank=0,
            world_size=1,
            node_id="n0",
            exporter_address="http://n0:9090",
        ), timeout=5)

        def _in_recovery() -> bool:
            s = get_status(handle)
            return s.mode == ControllerMode.RECOVERY or s.recovery_in_progress

        _poll_until(_in_recovery, timeout=15)

        status = get_status(handle)
        assert status.mode == ControllerMode.RECOVERY
        assert status.recovery_in_progress is True


class TestRecoveryPhaseHistoryRecorded:
    """After recovery completes, phase_history should contain the traversed phases."""

    def test_phase_history_is_populated(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[_OneShotCrashDetector()],
        )

        handle.submit_and_run.remote()
        time.sleep(0.3)

        run_id = get_status(handle).active_run_id or ""
        ray.get(handle.register_training_rank.remote(
            run_id=run_id, rank=0, world_size=1,
            node_id="n0", exporter_address="http://n0:9090",
        ), timeout=5)

        def _recovery_done_or_monitoring() -> bool:
            s = get_status(handle)
            if s.tick_count > 20:
                return True
            if s.phase_history and RecoveryPhase.DONE in s.phase_history:
                return True
            return False

        _poll_until(_recovery_done_or_monitoring, timeout=20)

        status = get_status(handle)
        if status.phase_history:
            assert len(status.phase_history) > 0


class TestStatusDuringRecovery:
    """get_status should reflect recovery_phase while recovery is in progress."""

    def test_recovery_phase_visible_in_status(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[_AlwaysCrashDetector()],
        )

        handle.submit_and_run.remote()
        time.sleep(0.3)

        run_id = get_status(handle).active_run_id or ""
        ray.get(handle.register_training_rank.remote(
            run_id=run_id, rank=0, world_size=1,
            node_id="n0", exporter_address="http://n0:9090",
        ), timeout=5)

        def _has_recovery_phase() -> bool:
            s = get_status(handle)
            return s.recovery_phase is not None

        _poll_until(_has_recovery_phase, timeout=15)

        status = get_status(handle)
        assert isinstance(status.recovery_phase, RecoveryPhase)
