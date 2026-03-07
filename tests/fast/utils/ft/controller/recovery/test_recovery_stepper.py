"""Tests for RecoveryStepper and RecoveryState classes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery.recovery_stepper import (
    DirectlyRestarting,
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryContext,
    RecoveryDone,
    RecoveryStepper,
    StopTimeDiagnostics,
)
from miles.utils.ft.controller.recovery.restart_stepper import (
    Evicting,
    MonitoringProgress,
    RestartDone,
    RestartFailed,
    RestartStepper,
    StoppingAndRestarting,
)
from miles.utils.ft.models.diagnostic import DiagnosticPipelineResult
from miles.utils.ft.models.fault import NodeFault, TriggerType
from miles.utils.ft.protocols.platform import JobStatus

from tests.fast.utils.ft.helpers.controller_fakes import (
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAlertChecker:
    """Programmable AlertChecker replacement."""

    def __init__(self, faults: list[NodeFault] | None = None) -> None:
        self._faults = faults or []

    def check_alerts(self) -> list[NodeFault]:
        return self._faults


class FakeDiagOrchestrator:
    """Returns a programmable DiagnosticPipelineResult."""

    def __init__(self, result: DiagnosticPipelineResult | None = None) -> None:
        self._result = result or DiagnosticPipelineResult()
        self.call_count: int = 0

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: object = None,
    ) -> DiagnosticPipelineResult:
        self.call_count += 1
        return self._result


def _make_restart_stepper(
    *,
    training_job: FakeTrainingJob | None = None,
    mini_wandb: MiniWandb | None = None,
    node_manager: FakeNodeManager | None = None,
    notifier: FakeNotifier | None = None,
) -> RestartStepper:
    return RestartStepper(
        node_manager=node_manager or FakeNodeManager(),
        training_job=training_job or FakeTrainingJob(),
        mini_wandb=mini_wandb or MiniWandb(),
        notifier=notifier,
        on_new_run=None,
        monitoring_success_iterations=10,
        monitoring_timeout_seconds=600,
    )


def _make_recovery_stepper(
    *,
    alert_checker: FakeAlertChecker | None = None,
    diagnostic_orchestrator: FakeDiagOrchestrator | None = None,
    restart_stepper: RestartStepper | None = None,
    notifier: FakeNotifier | None = None,
    timeout_seconds: int = 1800,
) -> RecoveryStepper:
    return RecoveryStepper(
        alert_checker=alert_checker or FakeAlertChecker(),
        diagnostic_orchestrator=diagnostic_orchestrator or FakeDiagOrchestrator(),
        restart_stepper=restart_stepper or _make_restart_stepper(),
        notifier=notifier,
        timeout_seconds=timeout_seconds,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ctx(
    *,
    trigger: TriggerType = TriggerType.CRASH,
    recovery_start_time: datetime | None = None,
) -> RecoveryContext:
    return RecoveryContext(
        trigger=trigger,
        recovery_start_time=recovery_start_time or _now(),
    )


async def _step(
    stepper: RecoveryStepper,
    state,
    *,
    trigger: TriggerType = TriggerType.CRASH,
    recovery_start_time: datetime | None = None,
):
    return await stepper(
        state,
        _ctx(trigger=trigger, recovery_start_time=recovery_start_time),
    )


# ---------------------------------------------------------------------------
# RealtimeChecks
# ---------------------------------------------------------------------------


class TestRealtimeChecks:
    @pytest.mark.asyncio
    async def test_no_faults_goes_to_directly_restarting(self) -> None:
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=[]))
        result = await _step(stepper, RealtimeChecks())
        assert isinstance(result, DirectlyRestarting)
        assert isinstance(result.restart, StoppingAndRestarting)

    @pytest.mark.asyncio
    async def test_non_ephemeral_faults_go_to_evicting_and_restarting(self) -> None:
        faults = [NodeFault(node_id="node-A", reason="gpu error", ephemeral=False)]
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=faults))
        result = await _step(stepper, RealtimeChecks())
        assert isinstance(result, EvictingAndRestarting)
        assert isinstance(result.restart, Evicting)
        assert result.restart.bad_node_ids == ["node-A"]
        assert result.is_final_attempt is False

    @pytest.mark.asyncio
    async def test_ephemeral_only_goes_to_directly_restarting(self) -> None:
        faults = [NodeFault(node_id="node-A", reason="temporary", ephemeral=True)]
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=faults))
        result = await _step(stepper, RealtimeChecks())
        assert isinstance(result, DirectlyRestarting)

    @pytest.mark.asyncio
    async def test_pre_identified_bad_nodes_skip_alert_check(self) -> None:
        stepper = _make_recovery_stepper()
        result = await _step(stepper, RealtimeChecks(pre_identified_bad_nodes=["node-X"]))
        assert isinstance(result, EvictingAndRestarting)
        assert result.restart.bad_node_ids == ["node-X"]

    @pytest.mark.asyncio
    async def test_multiple_bad_nodes_all_evicted(self) -> None:
        """Multiple non-ephemeral faults -> all bad nodes included in Evicting."""
        faults = [
            NodeFault(node_id="node-A", reason="gpu error", ephemeral=False),
            NodeFault(node_id="node-B", reason="network error", ephemeral=False),
            NodeFault(node_id="node-C", reason="cpu error", ephemeral=False),
        ]
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=faults))
        result = await _step(stepper, RealtimeChecks())

        assert isinstance(result, EvictingAndRestarting)
        assert result.restart.bad_node_ids == ["node-A", "node-B", "node-C"]

    @pytest.mark.asyncio
    async def test_ephemeral_faults_skip_eviction(self) -> None:
        """Only ephemeral faults -> DirectlyRestarting with StoppingAndRestarting (no Evicting)."""
        faults = [
            NodeFault(node_id="node-A", reason="temp glitch", ephemeral=True),
            NodeFault(node_id="node-B", reason="transient", ephemeral=True),
        ]
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=faults))
        result = await _step(stepper, RealtimeChecks())

        assert isinstance(result, DirectlyRestarting)
        assert isinstance(result.restart, StoppingAndRestarting)

    @pytest.mark.asyncio
    async def test_duplicate_bad_nodes_deduplicated(self) -> None:
        """Duplicate node IDs across faults are deduplicated via unique_node_ids."""
        faults = [
            NodeFault(node_id="node-A", reason="err1", ephemeral=False),
            NodeFault(node_id="node-A", reason="err2", ephemeral=False),
            NodeFault(node_id="node-B", reason="err3", ephemeral=False),
        ]
        stepper = _make_recovery_stepper(alert_checker=FakeAlertChecker(faults=faults))
        result = await _step(stepper, RealtimeChecks())

        assert isinstance(result, EvictingAndRestarting)
        assert result.restart.bad_node_ids == ["node-A", "node-B"]


# ---------------------------------------------------------------------------
# EvictingAndRestarting
# ---------------------------------------------------------------------------


class TestEvictingAndRestarting:
    @pytest.mark.asyncio
    async def test_restart_done_returns_recovery_done(self) -> None:
        restart_stepper = AsyncMock(return_value=RestartDone())
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = EvictingAndRestarting(restart=Evicting(bad_node_ids=["n"]))
        result = await _step(stepper, state)
        assert isinstance(result, RecoveryDone)

    @pytest.mark.asyncio
    async def test_restart_failed_non_final_goes_to_diagnostics(self) -> None:
        restart_stepper = AsyncMock(return_value=RestartFailed())
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = EvictingAndRestarting(restart=Evicting(), is_final_attempt=False)
        result = await _step(stepper, state)
        assert isinstance(result, StopTimeDiagnostics)

    @pytest.mark.asyncio
    async def test_restart_failed_final_goes_to_notify(self) -> None:
        restart_stepper = AsyncMock(return_value=RestartFailed())
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = EvictingAndRestarting(restart=Evicting(), is_final_attempt=True)
        result = await _step(stepper, state)
        assert isinstance(result, NotifyHumans)
        assert result.state_before == "EvictingAndRestarting"

    @pytest.mark.asyncio
    async def test_restart_in_progress_returns_updated_state(self) -> None:
        new_restart = StoppingAndRestarting(bad_node_ids=["n"], submitted=True)
        restart_stepper = AsyncMock(return_value=new_restart)
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = EvictingAndRestarting(restart=Evicting(bad_node_ids=["n"]))
        result = await _step(stepper, state)
        assert isinstance(result, EvictingAndRestarting)
        assert result.restart == new_restart

    @pytest.mark.asyncio
    async def test_restart_none_returns_none(self) -> None:
        restart_stepper = AsyncMock(return_value=None)
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = EvictingAndRestarting(restart=StoppingAndRestarting(submitted=True))
        result = await _step(stepper, state)
        assert result is None


# ---------------------------------------------------------------------------
# DirectlyRestarting
# ---------------------------------------------------------------------------


class TestDirectlyRestarting:
    @pytest.mark.asyncio
    async def test_restart_done_returns_recovery_done(self) -> None:
        restart_stepper = AsyncMock(return_value=RestartDone())
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = DirectlyRestarting(restart=StoppingAndRestarting())
        result = await _step(stepper, state)
        assert isinstance(result, RecoveryDone)

    @pytest.mark.asyncio
    async def test_restart_failed_goes_to_diagnostics(self) -> None:
        restart_stepper = AsyncMock(return_value=RestartFailed())
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)
        state = DirectlyRestarting(restart=StoppingAndRestarting())
        result = await _step(stepper, state)
        assert isinstance(result, StopTimeDiagnostics)


# ---------------------------------------------------------------------------
# StopTimeDiagnostics
# ---------------------------------------------------------------------------


class TestStopTimeDiagnostics:
    @pytest.mark.asyncio
    async def test_bad_nodes_found_goes_to_evicting_final(self) -> None:
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=["node-B"], reason="gpu fail"),
        )
        stepper = _make_recovery_stepper(diagnostic_orchestrator=diag)
        result = await _step(stepper, StopTimeDiagnostics())
        assert isinstance(result, EvictingAndRestarting)
        assert result.is_final_attempt is True
        assert result.restart.bad_node_ids == ["node-B"]

    @pytest.mark.asyncio
    async def test_no_bad_nodes_goes_to_notify(self) -> None:
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=[], reason="all passed"),
        )
        stepper = _make_recovery_stepper(diagnostic_orchestrator=diag)
        result = await _step(stepper, StopTimeDiagnostics())
        assert isinstance(result, NotifyHumans)
        assert result.state_before == "StopTimeDiagnostics"


# ---------------------------------------------------------------------------
# NotifyHumans
# ---------------------------------------------------------------------------


class TestNotifyHumans:
    @pytest.mark.asyncio
    async def test_notify_returns_recovery_done(self) -> None:
        notifier = FakeNotifier()
        stepper = _make_recovery_stepper(notifier=notifier)
        result = await _step(stepper, NotifyHumans(state_before="Test"))
        assert isinstance(result, RecoveryDone)
        assert len(notifier.calls) == 1
        assert "human intervention" in notifier.calls[0][1].lower()

    @pytest.mark.asyncio
    async def test_notify_humans_with_none_notifier_does_not_crash(self) -> None:
        stepper = _make_recovery_stepper(notifier=None)
        result = await _step(stepper, NotifyHumans(state_before="Test"))
        assert isinstance(result, RecoveryDone)


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------


class TestTerminal:
    @pytest.mark.asyncio
    async def test_recovery_done_is_terminal(self) -> None:
        stepper = _make_recovery_stepper()
        result = await _step(stepper, RecoveryDone())
        assert result is None


# ---------------------------------------------------------------------------
# Global timeout
# ---------------------------------------------------------------------------


class TestGlobalTimeout:
    @pytest.mark.asyncio
    async def test_timeout_forces_notify_humans(self) -> None:
        stepper = _make_recovery_stepper(timeout_seconds=60)
        old_time = _now() - timedelta(seconds=120)
        result = await stepper(
            RealtimeChecks(),
            _ctx(trigger=TriggerType.HANG, recovery_start_time=old_time),
        )
        assert isinstance(result, NotifyHumans)

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_notify_state(self) -> None:
        stepper = _make_recovery_stepper(timeout_seconds=60)
        old_time = _now() - timedelta(seconds=120)
        state = NotifyHumans(state_before="Test")
        result = await stepper(
            state,
            _ctx(trigger=TriggerType.HANG, recovery_start_time=old_time),
        )
        assert isinstance(result, RecoveryDone)

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_done_state(self) -> None:
        stepper = _make_recovery_stepper(timeout_seconds=60)
        old_time = _now() - timedelta(seconds=120)
        result = await stepper(
            RecoveryDone(),
            _ctx(trigger=TriggerType.HANG, recovery_start_time=old_time),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Full flow: DirectlyRestarting -> fail -> Diagnostics -> E&R -> Done
# ---------------------------------------------------------------------------


class TestFullRecoveryFlow:
    @pytest.mark.asyncio
    async def test_no_fault_direct_restart_success(self) -> None:
        """RealtimeChecks (no faults) -> DirectlyRestarting -> RestartDone -> RecoveryDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})

        restart_stepper = _make_restart_stepper(
            training_job=training_job, mini_wandb=mini_wandb,
        )
        stepper = _make_recovery_stepper(
            alert_checker=FakeAlertChecker(faults=[]),
            restart_stepper=restart_stepper,
        )

        # Step 1: RealtimeChecks -> DirectlyRestarting
        state = await _step(stepper, RealtimeChecks())
        assert isinstance(state, DirectlyRestarting)

        # Step 2: submit
        state = await _step(stepper, state)
        assert isinstance(state, DirectlyRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 3: poll -> MonitoringProgress
        state = await _step(stepper, state)
        assert isinstance(state, DirectlyRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 4: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await _step(stepper, state)
        assert isinstance(state, RecoveryDone)

    @pytest.mark.anyio
    async def test_fault_evict_restart_full_flow(self) -> None:
        """RealtimeChecks(pre_identified_bad_nodes) -> EvictingAndRestarting ->
        (evict, stop, restart, monitor) -> RestartDone -> RecoveryDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()

        restart_stepper = _make_restart_stepper(
            training_job=training_job, mini_wandb=mini_wandb,
            node_manager=node_manager, notifier=notifier,
        )
        stepper = _make_recovery_stepper(restart_stepper=restart_stepper)

        # Step 1: RealtimeChecks with pre-identified bad nodes -> EvictingAndRestarting
        state = await _step(stepper, RealtimeChecks(pre_identified_bad_nodes=["node-X"]))
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, Evicting)
        assert state.restart.bad_node_ids == ["node-X"]
        assert state.is_final_attempt is False

        # Step 2: Evicting -> mark node bad -> StoppingAndRestarting
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-X")

        # Step 3: submit
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 4: poll -> RUNNING -> MonitoringProgress
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 5: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await _step(stepper, state)
        assert isinstance(state, RecoveryDone)

    @pytest.mark.anyio
    async def test_direct_restart_fail_escalation_full_flow(self) -> None:
        """DirectlyRestarting fail -> StopTimeDiagnostics -> diagnostics find bad nodes ->
        EvictingAndRestarting (final attempt) -> RestartDone -> RecoveryDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})
        node_manager = FakeNodeManager()

        restart_stepper = _make_restart_stepper(
            training_job=training_job, mini_wandb=mini_wandb,
            node_manager=node_manager,
        )
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=["node-B"], reason="gpu fail"),
        )
        stepper = _make_recovery_stepper(
            alert_checker=FakeAlertChecker(faults=[]),
            restart_stepper=restart_stepper,
            diagnostic_orchestrator=diag,
        )

        # Step 1: RealtimeChecks (no faults) -> DirectlyRestarting
        state = await _step(stepper, RealtimeChecks())
        assert isinstance(state, DirectlyRestarting)

        # Step 2: submit
        state = await _step(stepper, state)
        assert isinstance(state, DirectlyRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 3: poll -> FAILED -> RestartFailed -> StopTimeDiagnostics
        state = await _step(stepper, state)
        assert isinstance(state, StopTimeDiagnostics)

        # Step 4: diagnostics find bad nodes -> EvictingAndRestarting (final attempt)
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert state.is_final_attempt is True
        assert state.restart.bad_node_ids == ["node-B"]
        assert diag.call_count == 1

        # Switch training job to succeed for the eviction restart path
        training_job._status_sequence = [JobStatus.RUNNING]

        # Step 5: Evicting -> mark node bad -> StoppingAndRestarting
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-B")

        # Step 6: submit
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 7: poll -> RUNNING -> MonitoringProgress
        state = await _step(stepper, state)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 8: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await _step(stepper, state)
        assert isinstance(state, RecoveryDone)

    @pytest.mark.asyncio
    async def test_notify_humans_then_done(self) -> None:
        """NotifyHumans -> RecoveryDone."""
        notifier = FakeNotifier()
        stepper = _make_recovery_stepper(notifier=notifier)
        result = await _step(stepper, NotifyHumans(state_before="Test"))
        assert isinstance(result, RecoveryDone)
