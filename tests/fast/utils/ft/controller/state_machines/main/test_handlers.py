"""Tests for main state machine handler classes."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import (
    AlwaysEnterRecoveryDetector,
    AlwaysNoneDetector,
    FakeNotifier,
    FixedDecisionDetector,
)
from tests.fast.utils.ft.utils.metric_injectors import make_detector_context

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main import (
    DetectingAnomaly,
    MainContext,
    Recovering,
    RestartingMainJob,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.recovery import (
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryDone,
    RecoveryEscalated,
)
from miles.utils.ft.controller.state_machines.restart import MonitoringProgress
from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stepper() -> StateMachineStepper:
    return create_main_stepper()


def _dummy_recovery_context_factory(trigger, recovery_start_time):
    """Dummy factory that returns a minimal object — only used when recovery_stepper is a mock."""
    return None


def _make_main_context(
    *,
    should_run_detectors: bool = True,
    detector_context: DetectorContext | None = None,
    rank_placement: dict[int, str] | None = None,
    detectors: list | None = None,
    cooldown: SlidingWindowThrottle | None = None,
    recovery_stepper: object | None = None,
    recovery_context_factory: object | None = None,
    on_recovery_duration: object | None = None,
    notifier: FakeNotifier | None = None,
    max_simultaneous_bad_nodes: int = 3,
    monitoring_config: MonitoringIterationProgressConfig | MonitoringSustainedAliveConfig | None = None,
    mini_wandb: MiniWandb | None = None,
) -> MainContext:
    return MainContext(
        job_status=JobStatus.RUNNING,
        tick_count=1,
        should_run_detectors=should_run_detectors,
        detector_context=(
            detector_context
            if detector_context is not None
            else (
                make_detector_context(
                    active_node_ids=(
                        set(rank_placement.values())
                        if rank_placement is not None
                        else {"node-0"}
                    ),
                )
                if should_run_detectors
                else None
            )
        ),
        notifier=notifier,
        detectors=detectors or [],
        cooldown=cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3),
        detector_crash_tracker=SlidingWindowCounter(window_seconds=1800, threshold=5),
        recovery_stepper=recovery_stepper or AsyncMock(return_value=None),
        recovery_context_factory=recovery_context_factory or _dummy_recovery_context_factory,
        on_recovery_duration=on_recovery_duration,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        monitoring_config=monitoring_config or MonitoringIterationProgressConfig(),
        mini_wandb=mini_wandb or MiniWandb(),
    )


# ---------------------------------------------------------------------------
# DetectingAnomaly
# ---------------------------------------------------------------------------


class TestDetectingAnomaly:
    @pytest.mark.asyncio
    async def test_no_detectors_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(DetectingAnomaly(), _make_main_context())
        assert result is None

    @pytest.mark.asyncio
    async def test_none_decision_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(detectors=[AlwaysNoneDetector()]),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_enter_recovery_transitions_to_recovering(self) -> None:
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(detectors=[AlwaysEnterRecoveryDetector()]),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert result.trigger == TriggerType.CRASH.value

    @pytest.mark.asyncio
    async def test_notify_human_sends_notification_stays_detecting(self) -> None:
        notifier = FakeNotifier()
        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="test notify",
                trigger=TriggerType.MISC,
            )
        )
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(detectors=[detector], notifier=notifier),
        )
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_cooldown_throttle_sends_notify(self) -> None:
        notifier = FakeNotifier()
        cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=1)
        cooldown.record()

        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[AlwaysEnterRecoveryDetector()],
                cooldown=cooldown,
                notifier=notifier,
            ),
        )
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_skip_detectors_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[AlwaysEnterRecoveryDetector()],
                should_run_detectors=False,
            ),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Template method filtering (BaseFaultDetector.evaluate)
# ---------------------------------------------------------------------------


class TestTemplateMethodFiltering:
    @pytest.mark.asyncio
    async def test_bad_node_not_in_rank_placement_skips_recovery(self) -> None:
        """Detector reports bad node that isn't in rank_placement -> filtered out, no recovery."""
        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-other"],
                reason="hw fault on inactive node",
                trigger=TriggerType.HARDWARE,
            )
        )
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[detector],
                rank_placement={0: "node-0"},
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_mixed_active_and_inactive_bad_nodes_filters(self) -> None:
        """Detector reports both active and inactive bad nodes -> only active ones kept."""
        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-0", "node-inactive"],
                reason="multi-node fault",
                trigger=TriggerType.HARDWARE,
            )
        )
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[detector],
                rank_placement={0: "node-0"},
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert result.recovery.pre_identified_bad_nodes == ["node-0"]

    @pytest.mark.asyncio
    async def test_detector_without_bad_node_ids_still_triggers_recovery(self) -> None:
        """Detector returning ENTER_RECOVERY with empty bad_node_ids is not affected by filtering."""
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(detectors=[AlwaysEnterRecoveryDetector()]),
        )
        assert isinstance(result, Recovering)

    @pytest.mark.asyncio
    async def test_inactive_node_does_not_block_subsequent_detectors(self) -> None:
        """First detector filtered out entirely -> second detector still runs and triggers recovery."""
        inactive_detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-inactive"],
                reason="inactive fault",
                trigger=TriggerType.HARDWARE,
            )
        )
        active_detector = AlwaysEnterRecoveryDetector(reason="active fault")
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[inactive_detector, active_detector],
                rank_placement={0: "node-0"},
            ),
        )
        assert isinstance(result, Recovering)


# ---------------------------------------------------------------------------
# Recovering
# ---------------------------------------------------------------------------


class TestRecovering:
    @pytest.mark.asyncio
    async def test_recovery_done_returns_detecting_anomaly(self) -> None:
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(return_value=RecoveryDone()),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, DetectingAnomaly)

    @pytest.mark.asyncio
    async def test_recovery_in_progress_stays_recovering(self) -> None:
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting, StopTimeDiagnostics
        from miles.utils.ft.controller.state_machines.restart import StoppingAndRestarting

        new_recovery = EvictingAndRestarting(
            restart=StoppingAndRestarting(),
            failed_next_state=StopTimeDiagnostics(),
        )
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(return_value=new_recovery),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, Recovering)
        assert result.recovery is new_recovery

    @pytest.mark.asyncio
    async def test_recovery_none_returns_none(self) -> None:
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(return_value=None),
                should_run_detectors=False,
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_recovery_exception_forces_notify(self) -> None:
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(side_effect=RuntimeError("boom")),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, NotifyHumans)

    @pytest.mark.anyio
    async def test_recovery_exception_forces_notify_then_done_then_detecting(self) -> None:
        """Exception -> NotifyHumans -> RecoveryDone -> DetectingAnomaly full chain."""
        recovery_stepper = AsyncMock(
            side_effect=[RuntimeError("boom"), RecoveryDone()],
        )
        stepper = _make_stepper()
        ctx = _make_main_context(
            recovery_stepper=recovery_stepper,
            should_run_detectors=False,
        )

        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )

        # Step 1: exception -> NotifyHumans
        result = await stepper(state, ctx)
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, NotifyHumans)

        # Step 2: NotifyHumans -> RecoveryDone -> DetectingAnomaly
        result = await stepper(result, ctx)
        assert isinstance(result, DetectingAnomaly)

    @pytest.mark.asyncio
    async def test_duration_callback_on_recovery_done(self) -> None:
        durations: list[float] = []
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(return_value=RecoveryDone()),
                on_recovery_duration=durations.append,
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, DetectingAnomaly)
        assert len(durations) == 1
        assert durations[0] >= 0

    @pytest.mark.asyncio
    async def test_dynamic_bad_nodes_resets_recovery(self) -> None:
        """Detector finds new bad nodes during recovery -> restart recovery with merged bad_node_ids."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting, StopTimeDiagnostics
        from miles.utils.ft.controller.state_machines.restart import Evicting

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="critical fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnostics(),
            ),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                detectors=[detector],
                recovery_stepper=AsyncMock(return_value=None),
                rank_placement={0: "node-old", 1: "node-new"},
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-old", "node-new"}


# ---------------------------------------------------------------------------
# StateMachine integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bad node count safeguard
# ---------------------------------------------------------------------------


class TestBadNodeCountSafeguard:
    @pytest.mark.asyncio
    async def test_too_many_bad_nodes_triggers_notify_human(self) -> None:
        """Detector reports >= threshold bad nodes -> NOTIFY_HUMAN, no recovery."""
        notifier = FakeNotifier()
        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-1", "node-2", "node-3"],
                reason="three nodes bad",
                trigger=TriggerType.HARDWARE,
            )
        )
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[detector],
                notifier=notifier,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-1", 1: "node-2", 2: "node-3"},
            ),
        )
        assert result is None
        assert len(notifier.calls) == 1
        assert "likely false positive" in notifier.calls[0][1]

    @pytest.mark.asyncio
    async def test_bad_nodes_below_threshold_enters_recovery(self) -> None:
        """Detector reports fewer bad nodes than threshold -> normal recovery."""
        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-1", "node-2"],
                reason="two nodes bad",
                trigger=TriggerType.HARDWARE,
            )
        )
        stepper = _make_stepper()
        result = await stepper(
            DetectingAnomaly(),
            _make_main_context(
                detectors=[detector],
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-1", 1: "node-2"},
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-1", "node-2"}

    @pytest.mark.asyncio
    async def test_too_many_dynamic_bad_nodes_continues_recovery(self) -> None:
        """Detectors report >= threshold new bad nodes during recovery -> NOTIFY_HUMAN but keep recovering."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting, StopTimeDiagnostics
        from miles.utils.ft.controller.state_machines.restart import Evicting

        notifier = FakeNotifier()

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-a", "node-b", "node-c"],
                reason="three critical faults",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnostics(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                detectors=[detector],
                recovery_stepper=AsyncMock(return_value=None),
                notifier=notifier,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old", 1: "node-a", 2: "node-b", 3: "node-c"},
            ),
        )
        assert result is None
        assert len(notifier.calls) == 1
        assert "likely false positive" in notifier.calls[0][1]

    @pytest.mark.asyncio
    async def test_already_known_bad_nodes_do_not_restart_recovery(self) -> None:
        """Detector re-reports nodes already being handled -> no recovery restart."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting, StopTimeDiagnostics
        from miles.utils.ft.controller.state_machines.restart import Evicting

        recovery_stepper = AsyncMock(return_value=None)

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-old"],
                reason="already known fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnostics(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                detectors=[detector],
                recovery_stepper=recovery_stepper,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old"},
            ),
        )
        assert result is None
        recovery_stepper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dynamic_bad_nodes_below_threshold_continues_recovery(self) -> None:
        """One new bad node during recovery (with 2 existing) -> normal merge, continues."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestarting, StopTimeDiagnostics
        from miles.utils.ft.controller.state_machines.restart import Evicting

        recovery_stepper = AsyncMock(return_value=None)

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="one critical fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = Recovering(
            recovery=EvictingAndRestarting(
                restart=Evicting(bad_node_ids=["node-old-1", "node-old-2"]),
                failed_next_state=StopTimeDiagnostics(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                detectors=[detector],
                recovery_stepper=recovery_stepper,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old-1", 1: "node-old-2", 2: "node-new"},
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, RealtimeChecks)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-old-1", "node-old-2", "node-new"}


# ---------------------------------------------------------------------------
# StateMachine integration
# ---------------------------------------------------------------------------


class TestInvalidDetectorDecision:
    @pytest.mark.anyio
    async def test_enter_recovery_without_trigger_raises(self) -> None:
        """ENTER_RECOVERY decision without trigger should raise ValueError."""
        malformed_decision = Decision.model_construct(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=[],
            reason="missing trigger",
            trigger=None,
        )
        detector = FixedDecisionDetector(malformed_decision)
        stepper = _make_stepper()
        with pytest.raises(ValueError, match="has no trigger"):
            await stepper(
                DetectingAnomaly(),
                _make_main_context(detectors=[detector]),
            )


class TestStateMachineIntegration:
    @pytest.mark.asyncio
    async def test_full_detecting_to_recovering_to_detecting(self) -> None:
        """DetectingAnomaly -> Recovering -> RecoveryDone -> DetectingAnomaly in one step()."""
        stepper = _make_stepper()
        ctx = _make_main_context(
            detectors=[AlwaysEnterRecoveryDetector()],
            recovery_stepper=AsyncMock(return_value=RecoveryDone()),
        )
        machine = StateMachine(initial_state=DetectingAnomaly(), stepper=stepper)

        await machine.step(ctx)
        assert isinstance(machine.state, DetectingAnomaly)
        assert len(machine.state_history) >= 2


# ---------------------------------------------------------------------------
# RecoveryEscalated -> RestartingMainJob
# ---------------------------------------------------------------------------


class TestRecoveryEscalated:
    @pytest.mark.asyncio
    async def test_recovery_escalated_transitions_to_restarting_main_job(self) -> None:
        """RecoveringHandler receives RecoveryEscalated -> returns RestartingMainJob."""
        stepper = _make_stepper()
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await stepper(
            state,
            _make_main_context(
                recovery_stepper=AsyncMock(return_value=RecoveryEscalated()),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, RestartingMainJob)


# ---------------------------------------------------------------------------
# RestartingMainJob handler (noop)
# ---------------------------------------------------------------------------


class TestRestartingMainJobHandler:
    @pytest.mark.asyncio
    async def test_noop_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(
            RestartingMainJob(),
            _make_main_context(should_run_detectors=False),
        )
        assert result is None


# ---------------------------------------------------------------------------
# RestartedMainJob handler
# ---------------------------------------------------------------------------


class TestRestartedMainJobHandler:
    @pytest.mark.asyncio
    async def test_creates_recovering_with_monitoring_progress(self) -> None:
        """RestartedMainJobHandler creates Recovering(EvictingAndRestarting(MonitoringProgress))."""
        from miles.utils.ft.controller.state_machines.main.models import RestartedMainJob

        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 42})

        stepper = _make_stepper()
        result = await stepper(
            RestartedMainJob(),
            _make_main_context(
                should_run_detectors=False,
                mini_wandb=mini_wandb,
                monitoring_config=MonitoringIterationProgressConfig(),
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert isinstance(result.recovery.restart, MonitoringProgress)
        assert result.recovery.restart.base_iteration == 42
        assert isinstance(result.recovery.failed_next_state, NotifyHumans)
        assert result.trigger == TriggerType.MISC

    @pytest.mark.asyncio
    async def test_sustained_alive_mode_uses_zero_base_iteration(self) -> None:
        """In sustained_alive mode, base_iteration is always 0."""
        from miles.utils.ft.controller.state_machines.main.models import RestartedMainJob

        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})

        stepper = _make_stepper()
        result = await stepper(
            RestartedMainJob(),
            _make_main_context(
                should_run_detectors=False,
                mini_wandb=mini_wandb,
                monitoring_config=MonitoringSustainedAliveConfig(),
            ),
        )
        assert isinstance(result, Recovering)
        assert isinstance(result.recovery, EvictingAndRestarting)
        assert isinstance(result.recovery.restart, MonitoringProgress)
        assert result.recovery.restart.base_iteration == 0
