"""Tests for subsystem state machine handler classes."""

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
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import MetricStore
from miles.utils.ft.controller.state_machines.subsystem import (
    DetectingAnomalySt,
    SubsystemContext,
    RecoveringSt,
    create_subsystem_stepper,
)
from miles.utils.ft.controller.state_machines.recovery import (
    EvictingAndRestartingSt,
    NotifyHumansSt,
    RealtimeChecksSt,
    RecoveryDoneSt,
)
from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter, SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stepper() -> StateMachineStepper:
    return create_subsystem_stepper()


def _dummy_recovery_context_factory(trigger, recovery_start_time):
    """Dummy factory that returns a minimal object — only used when recovery_stepper is a mock."""
    return None


def _make_subsystem_context(
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
    metric_store: MetricStore | None = None,
) -> SubsystemContext:
    return SubsystemContext(
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
        recovery_stepper=recovery_stepper or _mock_stepper_yielding(None),
        recovery_context_factory=recovery_context_factory or _dummy_recovery_context_factory,
        on_recovery_duration=on_recovery_duration,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        monitoring_config=monitoring_config or MonitoringIterationProgressConfig(),
        metric_store=metric_store or MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
    )


async def _step_last(stepper, state, ctx):
    result = None
    async for result in stepper(state, ctx):
        pass
    return result


def _mock_stepper_yielding(value: object):
    """Create a fake stepper (async generator) that yields a single value, or nothing if None."""
    async def _stepper(state: object, ctx: object):
        _stepper.call_count += 1
        if value is not None:
            yield value
    _stepper.call_count = 0
    return _stepper


def _mock_stepper_raising(exc: BaseException):
    """Create a fake stepper (async generator) that raises an exception."""
    async def _stepper(state: object, ctx: object):
        raise exc
        yield  # unreachable but makes this an async generator
    return _stepper


def _mock_stepper_sequence(effects: list):
    """Create a fake stepper that yields/raises from a sequence of effects (one per call)."""
    idx = [0]
    async def _stepper(state: object, ctx: object):
        i = min(idx[0], len(effects) - 1)
        idx[0] += 1
        effect = effects[i]
        if isinstance(effect, BaseException):
            raise effect
        if effect is not None:
            yield effect
    return _stepper


# ---------------------------------------------------------------------------
# DetectingAnomaly
# ---------------------------------------------------------------------------


class TestDetectingAnomaly:
    @pytest.mark.asyncio
    async def test_no_detectors_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper, DetectingAnomalySt(), _make_subsystem_context())
        assert result is None

    @pytest.mark.asyncio
    async def test_none_decision_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(detectors=[AlwaysNoneDetector()]),
                                  )
        assert result is None

    @pytest.mark.asyncio
    async def test_enter_recovery_transitions_to_recovering(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(detectors=[AlwaysEnterRecoveryDetector()]),
                                  )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(detectors=[detector], notifier=notifier),
                                  )
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_cooldown_throttle_sends_notify(self) -> None:
        notifier = FakeNotifier()
        cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=1)
        cooldown.record()

        stepper = _make_stepper()
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
                detectors=[AlwaysEnterRecoveryDetector()],
                cooldown=cooldown,
                notifier=notifier,
            ),
                                  )
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_throttled_recovery_does_not_consume_cooldown(self) -> None:
        """Previously cooldown.record() was called before is_throttled(),
        so even throttled attempts consumed a cooldown slot. With max_count=2,
        only 1 actual recovery would succeed because the 2nd slot was wasted
        on a throttled attempt. Now record() is called only when recovery
        actually proceeds."""
        cooldown = SlidingWindowThrottle(window_minutes=30.0, max_count=2)
        cooldown.record()
        assert cooldown.is_throttled() is False

        notifier = FakeNotifier()
        cooldown_before_throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=2)
        cooldown_before_throttle.record()
        cooldown_before_throttle.record()
        assert cooldown_before_throttle.is_throttled() is True

        stepper = _make_stepper()
        result = await _step_last(
            stepper,
            DetectingAnomalySt(),
            _make_subsystem_context(
                detectors=[AlwaysEnterRecoveryDetector()],
                cooldown=cooldown_before_throttle,
                notifier=notifier,
            ),
        )
        assert result is None
        assert len(notifier.calls) == 1

    @pytest.mark.asyncio
    async def test_skip_detectors_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
                detectors=[detector],
                rank_placement={0: "node-0"},
            ),
                                  )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
        assert result.recovery.pre_identified_bad_nodes == ("node-0",)

    @pytest.mark.asyncio
    async def test_detector_without_bad_node_ids_still_triggers_recovery(self) -> None:
        """Detector returning ENTER_RECOVERY with empty bad_node_ids is not affected by filtering."""
        stepper = _make_stepper()
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(detectors=[AlwaysEnterRecoveryDetector()]),
                                  )
        assert isinstance(result, RecoveringSt)

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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
                detectors=[inactive_detector, active_detector],
                rank_placement={0: "node-0"},
            ),
                                  )
        assert isinstance(result, RecoveringSt)


# ---------------------------------------------------------------------------
# Recovering
# ---------------------------------------------------------------------------


class TestRecovering:
    @pytest.mark.asyncio
    async def test_recovery_done_returns_detecting_anomaly(self) -> None:
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                recovery_stepper=_mock_stepper_yielding(RecoveryDoneSt()),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, DetectingAnomalySt)

    @pytest.mark.asyncio
    async def test_recovery_in_progress_stays_recovering(self) -> None:
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import StoppingAndRestartingSt

        new_recovery = EvictingAndRestartingSt(
            restart=StoppingAndRestartingSt(),
            failed_next_state=StopTimeDiagnosticsSt(),
        )
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                recovery_stepper=_mock_stepper_yielding(new_recovery),
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert result.recovery is new_recovery

    @pytest.mark.asyncio
    async def test_recovery_none_returns_none(self) -> None:
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                recovery_stepper=_mock_stepper_yielding(None),
                should_run_detectors=False,
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_recovery_exception_propagates_to_tick_loop(self) -> None:
        """Previously _advance_recovery caught all exceptions and funneled them
        into NotifyHumansSt, masking bugs in handler code. Now exceptions
        propagate to the TickLoop tick_failure_tracker so they surface as
        tick failures (not infinite recovery loops).
        """
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        with pytest.raises(RuntimeError, match="boom"):
            await _step_last(stepper,
                state,
                _make_subsystem_context(
                    recovery_stepper=_mock_stepper_raising(RuntimeError("boom")),
                    should_run_detectors=False,
                ),
            )

    @pytest.mark.asyncio
    async def test_duration_callback_on_recovery_done(self) -> None:
        durations: list[float] = []
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                recovery_stepper=_mock_stepper_yielding(RecoveryDoneSt()),
                on_recovery_duration=durations.append,
                should_run_detectors=False,
            ),
        )
        assert isinstance(result, DetectingAnomalySt)
        assert len(durations) == 1
        assert durations[0] >= 0

    @pytest.mark.asyncio
    async def test_dynamic_bad_nodes_resets_recovery(self) -> None:
        """Detector finds new bad nodes during recovery -> restart recovery with merged bad_node_ids."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="critical fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["node-old"],
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=_mock_stepper_yielding(None),
                rank_placement={0: "node-old", 1: "node-new"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-old", "node-new"}

    @pytest.mark.asyncio
    async def test_new_bad_nodes_during_eviction_restarts_from_realtime_checks(self) -> None:
        """When new bad nodes appear mid-eviction, recovery restarts from
        RealtimeChecksSt with the merged set. This is intentional: partial
        eviction may be based on incomplete fault information, so re-entering
        RealtimeChecks ensures the full set goes through the pipeline."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="new fault during eviction",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["node-old"],
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=_mock_stepper_yielding(None),
                rank_placement={0: "node-old", 1: "node-new"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
        assert set(result.known_bad_node_ids) == {"node-old", "node-new"}

    @pytest.mark.asyncio
    async def test_cascading_bad_node_preserves_original_recovery_start_time(self) -> None:
        """Previously _check_new_bad_nodes used datetime.now() as the new
        recovery_start_time, resetting the global timeout clock. This meant
        cascading failures could extend recovery indefinitely. Now the
        original recovery_start_time is preserved.
        """
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="cascading fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        original_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH.value,
            recovery_start_time=original_start,
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=_mock_stepper_yielding(None),
                rank_placement={0: "node-old", 1: "node-new"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert result.recovery_start_time == original_start


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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
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
        result = await _step_last(stepper,
                                  DetectingAnomalySt(),
                                  _make_subsystem_context(
                detectors=[detector],
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-1", 1: "node-2"},
            ),
                                  )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-1", "node-2"}

    @pytest.mark.asyncio
    async def test_too_many_dynamic_bad_nodes_transitions_to_notify_humans(self) -> None:
        """M-4: Detectors report >= threshold new bad nodes during recovery ->
        transition to NotifyHumansSt instead of inline notification (previously
        notify_too_many_bad_nodes fired every tick with no state change)."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-a", "node-b", "node-c"],
                reason="three critical faults",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=_mock_stepper_yielding(None),
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old", 1: "node-a", 2: "node-b", 3: "node-c"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, NotifyHumansSt)
        assert result.recovery.reason == "too_many_simultaneous_bad_nodes"

    @pytest.mark.asyncio
    async def test_already_known_bad_nodes_do_not_restart_recovery(self) -> None:
        """Detector re-reports nodes already being handled -> no recovery restart."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        recovery_stepper = _mock_stepper_yielding(None)

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-old"],
                reason="already known fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["node-old"],
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=recovery_stepper,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old"},
            ),
        )
        assert result is None
        assert recovery_stepper.call_count == 1

    @pytest.mark.asyncio
    async def test_dynamic_bad_nodes_below_threshold_continues_recovery(self) -> None:
        """One new bad node during recovery (with 2 existing) -> normal merge, continues."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        recovery_stepper = _mock_stepper_yielding(None)

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-new"],
                reason="one critical fault",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old-1", "node-old-2"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=["node-old-1", "node-old-2"],
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=recovery_stepper,
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old-1", 1: "node-old-2", 2: "node-new"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, RealtimeChecksSt)
        assert set(result.recovery.pre_identified_bad_nodes) == {"node-old-1", "node-old-2", "node-new"}
        assert set(result.known_bad_node_ids) == {"node-old-1", "node-old-2", "node-new"}

    @pytest.mark.asyncio
    async def test_too_many_bad_nodes_includes_merged_set_in_notify_humans(self) -> None:
        """Previously when too many new bad nodes triggered NotifyHumansSt,
        the newly discovered bad nodes were lost — known_bad_node_ids stayed
        at the old value and NotifyHumansSt had no bad node info. Now the
        merged set (known + new) is stored in both NotifyHumansSt.bad_node_ids
        and RecoveringSt.known_bad_node_ids."""
        from miles.utils.ft.controller.state_machines.recovery import EvictingAndRestartingSt, StopTimeDiagnosticsSt
        from miles.utils.ft.controller.state_machines.restart import EvictingSt

        detector = FixedDecisionDetector(
            Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=["node-a", "node-b", "node-c"],
                reason="three critical faults",
                trigger=TriggerType.HARDWARE,
            )
        )

        stepper = _make_stepper()
        state = RecoveringSt(
            recovery=EvictingAndRestartingSt(
                restart=EvictingSt(bad_node_ids=["node-old"]),
                failed_next_state=StopTimeDiagnosticsSt(),
            ),
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
            known_bad_node_ids=("node-old",),
        )
        result = await _step_last(stepper,
            state,
            _make_subsystem_context(
                detectors=[detector],
                recovery_stepper=_mock_stepper_yielding(None),
                max_simultaneous_bad_nodes=3,
                rank_placement={0: "node-old", 1: "node-a", 2: "node-b", 3: "node-c"},
            ),
        )
        assert isinstance(result, RecoveringSt)
        assert isinstance(result.recovery, NotifyHumansSt)

        # Step 1: NotifyHumansSt carries the merged bad node set
        expected_all = {"node-a", "node-b", "node-c", "node-old"}
        assert set(result.recovery.bad_node_ids) == expected_all

        # Step 2: known_bad_node_ids on RecoveringSt is also updated
        assert set(result.known_bad_node_ids) == expected_all


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
            await _step_last(stepper,
                             DetectingAnomalySt(),
                             _make_subsystem_context(detectors=[detector]),
                             )


class TestStateMachineIntegration:
    @pytest.mark.asyncio
    async def test_full_detecting_to_recovering_to_detecting(self) -> None:
        """DetectingAnomaly -> Recovering -> RecoveryDone -> DetectingAnomaly in one step()."""
        stepper = _make_stepper()
        ctx = _make_subsystem_context(
            detectors=[AlwaysEnterRecoveryDetector()],
            recovery_stepper=_mock_stepper_yielding(RecoveryDoneSt()),
        )
        machine = StateMachine(initial_state=DetectingAnomalySt(), stepper=stepper)

        await machine.step(ctx)
        assert isinstance(machine.state, DetectingAnomalySt)
        assert len(machine.state_history) >= 2


