"""Tests for MainStepper and MainState classes."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.main_state_machine import (
    DetectingAnomaly,
    MainStepper,
    Recovering,
)
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.controller.recovery.recovery_stepper import (
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    RecoveryStepper,
)
from miles.utils.ft.controller.recovery.restart_stepper import RestartStepper
from miles.utils.ft.controller.state_machine import StateMachine
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus

from tests.fast.utils.ft.helpers.controller_fakes import (
    AlwaysEnterRecoveryDetector,
    AlwaysNoneDetector,
    CriticalFixedDecisionDetector,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    FixedDecisionDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_platform_deps(*, notifier: FakeNotifier | None = None):
    from miles.utils.ft.controller.actions import PlatformDeps
    from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig

    return PlatformDeps(
        node_manager=FakeNodeManager(),
        training_job=FakeTrainingJob(),
        metric_store=MiniPrometheus(config=MiniPrometheusConfig()),
        mini_wandb=MiniWandb(),
        notifier=notifier,
        diagnostic_orchestrator=AsyncMock(),
        controller_exporter=None,
    )


def _make_detector_context() -> DetectorContext:
    from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig

    return DetectorContext(
        metric_store=MiniPrometheus(config=MiniPrometheusConfig()),
        mini_wandb=MiniWandb(),
        rank_placement={0: "node-0"},
        job_status=JobStatus.RUNNING,
    )


def _make_main_stepper(
    *,
    detectors=None,
    cooldown: SlidingWindowThrottle | None = None,
    recovery_stepper: RecoveryStepper | AsyncMock | None = None,
    on_recovery_duration=None,
    notifier: FakeNotifier | None = None,
) -> MainStepper:
    from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
    from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig

    restart_stepper = RestartStepper(
        node_manager=FakeNodeManager(),
        training_job=FakeTrainingJob(),
        mini_wandb=MiniWandb(),
        notifier=None,
        on_new_run=None,
        monitoring_success_iterations=10,
        monitoring_timeout_seconds=600,
    )

    if recovery_stepper is None:
        recovery_stepper = RecoveryStepper(
            alert_checker=AlertChecker(metric_store=MiniPrometheus(config=MiniPrometheusConfig())),
            diagnostic_orchestrator=AsyncMock(),
            restart_stepper=restart_stepper,
            notifier=None,
        )

    return MainStepper(
        platform_deps=_make_platform_deps(notifier=notifier),
        recovery_stepper=recovery_stepper,
        detectors=detectors or [],
        cooldown=cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3),
        on_recovery_duration=on_recovery_duration,
    )


# ---------------------------------------------------------------------------
# DetectingAnomaly
# ---------------------------------------------------------------------------


class TestDetectingAnomaly:
    @pytest.mark.asyncio
    async def test_no_detectors_returns_none(self) -> None:
        stepper = _make_main_stepper()
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )
        result = await stepper(DetectingAnomaly())
        assert result is None

    @pytest.mark.asyncio
    async def test_none_decision_returns_none(self) -> None:
        stepper = _make_main_stepper(detectors=[AlwaysNoneDetector()])
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )
        result = await stepper(DetectingAnomaly())
        assert result is None

    @pytest.mark.asyncio
    async def test_enter_recovery_transitions_to_recovering(self) -> None:
        stepper = _make_main_stepper(detectors=[AlwaysEnterRecoveryDetector()])
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )
        result = await stepper(DetectingAnomaly())
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert result.trigger == TriggerType.CRASH.value

    @pytest.mark.asyncio
    async def test_notify_human_sends_notification_stays_detecting(self) -> None:
        notifier = FakeNotifier()
        detector = FixedDecisionDetector(Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="test notify",
            trigger=TriggerType.MISC,
        ))
        stepper = _make_main_stepper(detectors=[detector], notifier=notifier)
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )
        result = await stepper(DetectingAnomaly())
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_cooldown_throttle_sends_notify(self) -> None:
        notifier = FakeNotifier()
        cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=1)
        cooldown.record(TriggerType.CRASH)

        stepper = _make_main_stepper(
            detectors=[AlwaysEnterRecoveryDetector()],
            cooldown=cooldown,
            notifier=notifier,
        )
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )
        result = await stepper(DetectingAnomaly())
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_skip_detectors_returns_none(self) -> None:
        stepper = _make_main_stepper(detectors=[AlwaysEnterRecoveryDetector()])
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )
        result = await stepper(DetectingAnomaly())
        assert result is None


# ---------------------------------------------------------------------------
# Recovering
# ---------------------------------------------------------------------------


class TestRecovering:
    @pytest.mark.asyncio
    async def test_recovery_done_returns_detecting_anomaly(self) -> None:
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=RecoveryDone())
        stepper = _make_main_stepper(recovery_stepper=recovery_stepper)
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(state)
        assert isinstance(result, DetectingAnomaly)

    @pytest.mark.asyncio
    async def test_recovery_in_progress_stays_recovering(self) -> None:
        from miles.utils.ft.controller.recovery.recovery_stepper import DirectlyRestarting
        from miles.utils.ft.controller.recovery.restart_stepper import StoppingAndRestarting

        new_recovery = DirectlyRestarting(restart=StoppingAndRestarting())
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=new_recovery)
        stepper = _make_main_stepper(recovery_stepper=recovery_stepper)
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(state)
        assert isinstance(result, Recovering)
        assert result.recovery is new_recovery

    @pytest.mark.asyncio
    async def test_recovery_none_returns_none(self) -> None:
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=None)
        stepper = _make_main_stepper(recovery_stepper=recovery_stepper)
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(state)
        assert result is None

    @pytest.mark.asyncio
    async def test_recovery_exception_forces_notify(self) -> None:
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(side_effect=RuntimeError("boom"))
        stepper = _make_main_stepper(recovery_stepper=recovery_stepper)
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(state)
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, NotifyHumans)

    @pytest.mark.asyncio
    async def test_duration_callback_on_recovery_done(self) -> None:
        durations: list[float] = []
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=RecoveryDone())
        stepper = _make_main_stepper(
            recovery_stepper=recovery_stepper,
            on_recovery_duration=durations.append,
        )
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=False,
            detector_context=None,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        await stepper(state)
        assert len(durations) == 1
        assert durations[0] >= 0

    @pytest.mark.asyncio
    async def test_dynamic_bad_nodes_resets_recovery(self) -> None:
        """Critical detector finds new bad nodes -> restart recovery with merged bad_node_ids."""
        from miles.utils.ft.controller.recovery.recovery_stepper import EvictingAndRestarting
        from miles.utils.ft.controller.recovery.restart_stepper import Evicting

        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=None)

        critical_detector = CriticalFixedDecisionDetector(Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-new"],
            reason="critical fault",
            trigger=TriggerType.HARDWARE,
        ))

        stepper = _make_main_stepper(
            detectors=[critical_detector],
            recovery_stepper=recovery_stepper,
        )
        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )

        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-old"]),
            ),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(state)
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-old", "node-new"}


# ---------------------------------------------------------------------------
# StateMachine integration
# ---------------------------------------------------------------------------


class TestStateMachineIntegration:
    @pytest.mark.asyncio
    async def test_full_detecting_to_recovering_to_detecting(self) -> None:
        """DetectingAnomaly -> Recovering -> RecoveryDone -> DetectingAnomaly in one step()."""
        recovery_stepper = AsyncMock()
        recovery_stepper.step_with_context = AsyncMock(return_value=RecoveryDone())

        stepper = _make_main_stepper(
            detectors=[AlwaysEnterRecoveryDetector()],
            recovery_stepper=recovery_stepper,
        )
        machine = StateMachine(initial_state=DetectingAnomaly(), stepper=stepper)

        stepper.set_tick_context(
            job_status=JobStatus.RUNNING,
            tick_count=1,
            should_run_detectors=True,
            detector_context=_make_detector_context(),
        )

        await machine.step()
        assert isinstance(machine.state, DetectingAnomaly)
        assert len(machine.state_history) >= 2
