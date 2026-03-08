"""Local Ray: Recovery flow — trigger recovery via injected detector, observe state transitions."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import get_status, poll_for_run_id
from tests.fast.utils.ft.utils.controller_fakes import OneShotCrashDetector

from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.types import ActionType, ControllerMode, Decision, TriggerType

pytestmark = [
    pytest.mark.local_ray,
]


class _AlwaysCrashDetector(BaseFaultDetector):
    """Detector that fires ENTER_RECOVERY on every evaluation."""

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason="injected crash for test",
            trigger=TriggerType.CRASH,
        )


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
        run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

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
            detectors_override=[OneShotCrashDetector()],
            monitoring_success_iterations_override=0,
        )

        handle.submit_and_run.remote()
        run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        def _recovery_done_or_monitoring() -> bool:
            s = get_status(handle)
            if s.tick_count > 20:
                return True
            if s.phase_history and "RecoveryDone" in s.phase_history:
                return True
            return False

        _poll_until(_recovery_done_or_monitoring, timeout=20)

        status = get_status(handle)
        assert status.phase_history is not None, "phase_history should be populated after recovery"
        assert any(
            p in {"RealtimeChecks", "StopTimeDiagnostics", "RecoveryDone", "NotifyHumans"}
            for p in status.phase_history
        ), f"phase_history should contain recovery phases, got: {status.phase_history}"


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
        run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        def _has_recovery_phase() -> bool:
            s = get_status(handle)
            return s.recovery_phase is not None

        _poll_until(_has_recovery_phase, timeout=15)

        status = get_status(handle)
        assert isinstance(status.recovery_phase, str)


class TestControllerKilledDuringRecovery:
    """Kill controller while recovery is in progress → restarts with fresh state (R3)."""

    def test_kill_during_recovery_resets_to_monitoring(
        self,
        make_controller_actor: Callable[..., ray.actor.ActorHandle],
    ) -> None:
        handle = make_controller_actor(
            detectors_override=[_AlwaysCrashDetector()],
        )

        handle.submit_and_run.remote()
        run_id = poll_for_run_id(handle)

        ray.get(
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        def _in_recovery() -> bool:
            s = get_status(handle)
            return s.mode == ControllerMode.RECOVERY

        _poll_until(_in_recovery, timeout=15)

        ray.kill(handle, no_restart=False)

        name = ft_controller_actor_name("")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                restarted = ray.get_actor(name)
                status = ray.get(restarted.get_status.remote(), timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise TimeoutError("Actor did not restart within 10s")

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False
