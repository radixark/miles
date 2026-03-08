"""Tests for recovery stepper handler classes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.ft.utils.controller_fakes import FakeNodeManager, FakeNotifier, FakeTrainingJob

from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.recovery import (
    EvictingAndRestarting,
    NotifyHumans,
    RealtimeChecks,
    RecoveryContext,
    RecoveryDone,
    StopTimeDiagnostics,
    create_recovery_stepper,
)
from miles.utils.ft.controller.state_machines.restart import (
    Evicting,
    MonitoringProgress,
    RestartContext,
    RestartDone,
    RestartFailed,
    StoppingAndRestarting,
    create_restart_stepper,
)
from miles.utils.ft.models.diagnostic import DiagnosticPipelineResult
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.protocols.platform import JobStatus
from miles.utils.ft.utils.state_machine import StateMachineStepper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDiagOrchestrator:
    """Returns a programmable DiagnosticPipelineResult."""

    def __init__(self, result: DiagnosticPipelineResult | None = None) -> None:
        self._result = result or DiagnosticPipelineResult()
        self.call_count: int = 0

    async def run_diagnostic_pipeline(
        self,
        pre_executors: object = None,
    ) -> DiagnosticPipelineResult:
        self.call_count += 1
        return self._result


def _make_stepper(*, timeout_seconds: int = 1800) -> StateMachineStepper:
    return create_recovery_stepper()


def _make_restart_stepper_and_context(
    *,
    training_job: FakeTrainingJob | None = None,
    mini_wandb: MiniWandb | None = None,
    node_manager: FakeNodeManager | None = None,
    notifier: FakeNotifier | None = None,
) -> tuple[StateMachineStepper, RestartContext]:
    resolved_node_manager = node_manager or FakeNodeManager()
    resolved_training_job = training_job or FakeTrainingJob()
    resolved_mini_wandb = mini_wandb or MiniWandb()

    stepper = create_restart_stepper()
    ctx = RestartContext(
        node_manager=resolved_node_manager,
        training_job=resolved_training_job,
        mini_wandb=resolved_mini_wandb,
        notifier=notifier,
        on_new_run=None,
        monitoring_success_iterations=10,
        monitoring_timeout_seconds=600,
    )
    return stepper, ctx


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_ctx(
    *,
    trigger: TriggerType = TriggerType.CRASH,
    recovery_start_time: datetime | None = None,
    diagnostic_orchestrator: FakeDiagOrchestrator | None = None,
    restart_stepper: object | None = None,
    restart_context: RestartContext | None = None,
    notifier: FakeNotifier | None = None,
    timeout_seconds: int = 1800,
    rank_pids_provider: object | None = None,
) -> RecoveryContext:
    if restart_stepper is None and restart_context is None:
        real_stepper, real_ctx = _make_restart_stepper_and_context()
        restart_stepper = real_stepper
        restart_context = real_ctx
    elif restart_stepper is None:
        restart_stepper = create_restart_stepper()
    elif restart_context is None:
        _, restart_context = _make_restart_stepper_and_context()

    return RecoveryContext(
        trigger=trigger,
        recovery_start_time=recovery_start_time or _now(),
        diagnostic_orchestrator=diagnostic_orchestrator or FakeDiagOrchestrator(),
        restart_stepper=restart_stepper,
        restart_context=restart_context,
        notifier=notifier,
        timeout_seconds=timeout_seconds,
        rank_pids_provider=rank_pids_provider,
    )


async def _step(
    stepper: StateMachineStepper,
    state: object,
    *,
    ctx: RecoveryContext | None = None,
    trigger: TriggerType = TriggerType.CRASH,
    recovery_start_time: datetime | None = None,
    **ctx_kwargs: object,
) -> object:
    if ctx is None:
        ctx = _make_ctx(
            trigger=trigger,
            recovery_start_time=recovery_start_time,
            **ctx_kwargs,
        )
    return await stepper(state, ctx)


# ---------------------------------------------------------------------------
# RealtimeChecks
# ---------------------------------------------------------------------------


class TestRealtimeChecks:
    @pytest.mark.asyncio
    async def test_no_pre_identified_goes_to_direct_restart(self) -> None:
        stepper = _make_stepper()
        result = await _step(stepper, RealtimeChecks())
        assert isinstance(result, EvictingAndRestarting)
        assert isinstance(result.restart, StoppingAndRestarting)
        assert isinstance(result.failed_next_state, StopTimeDiagnostics)

    @pytest.mark.asyncio
    async def test_pre_identified_bad_nodes_go_to_evicting(self) -> None:
        stepper = _make_stepper()
        result = await _step(stepper, RealtimeChecks(pre_identified_bad_nodes=["node-X"]))
        assert isinstance(result, EvictingAndRestarting)
        assert isinstance(result.restart, Evicting)
        assert result.restart.bad_node_ids == ["node-X"]

    @pytest.mark.asyncio
    async def test_multiple_pre_identified_bad_nodes_all_evicted(self) -> None:
        stepper = _make_stepper()
        result = await _step(
            stepper,
            RealtimeChecks(pre_identified_bad_nodes=["node-A", "node-B", "node-C"]),
        )
        assert isinstance(result, EvictingAndRestarting)
        assert result.restart.bad_node_ids == ["node-A", "node-B", "node-C"]


# ---------------------------------------------------------------------------
# EvictingAndRestarting
# ---------------------------------------------------------------------------


class TestEvictingAndRestarting:
    @pytest.mark.asyncio
    async def test_restart_done_returns_recovery_done(self) -> None:
        stepper = _make_stepper()
        ctx = _make_ctx(restart_stepper=AsyncMock(return_value=RestartDone()))
        state = EvictingAndRestarting(
            restart=Evicting(bad_node_ids=["n"]),
            failed_next_state=StopTimeDiagnostics(),
        )
        result = await stepper(state, ctx)
        assert isinstance(result, RecoveryDone)

    @pytest.mark.asyncio
    async def test_restart_failed_returns_failed_next_state(self) -> None:
        stepper = _make_stepper()
        ctx = _make_ctx(restart_stepper=AsyncMock(return_value=RestartFailed()))
        state = EvictingAndRestarting(
            restart=Evicting(),
            failed_next_state=StopTimeDiagnostics(),
        )
        result = await stepper(state, ctx)
        assert isinstance(result, StopTimeDiagnostics)

    @pytest.mark.asyncio
    async def test_restart_failed_with_notify_next_state(self) -> None:
        stepper = _make_stepper()
        ctx = _make_ctx(restart_stepper=AsyncMock(return_value=RestartFailed()))
        state = EvictingAndRestarting(
            restart=Evicting(),
            failed_next_state=NotifyHumans(state_before="EvictingAndRestarting"),
        )
        result = await stepper(state, ctx)
        assert isinstance(result, NotifyHumans)
        assert result.state_before == "EvictingAndRestarting"

    @pytest.mark.asyncio
    async def test_restart_in_progress_returns_updated_state(self) -> None:
        new_restart = StoppingAndRestarting(bad_node_ids=["n"], submitted=True)
        stepper = _make_stepper()
        ctx = _make_ctx(restart_stepper=AsyncMock(return_value=new_restart))
        state = EvictingAndRestarting(
            restart=Evicting(bad_node_ids=["n"]),
            failed_next_state=StopTimeDiagnostics(),
        )
        result = await stepper(state, ctx)
        assert isinstance(result, EvictingAndRestarting)
        assert result.restart == new_restart
        assert isinstance(result.failed_next_state, StopTimeDiagnostics)

    @pytest.mark.asyncio
    async def test_restart_none_returns_none(self) -> None:
        stepper = _make_stepper()
        ctx = _make_ctx(restart_stepper=AsyncMock(return_value=None))
        state = EvictingAndRestarting(
            restart=StoppingAndRestarting(submitted=True),
            failed_next_state=StopTimeDiagnostics(),
        )
        result = await stepper(state, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# StopTimeDiagnostics
# ---------------------------------------------------------------------------


class TestStopTimeDiagnostics:
    @pytest.mark.asyncio
    async def test_bad_nodes_found_goes_to_evicting_with_notify_on_fail(self) -> None:
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=["node-B"], reason="gpu fail"),
        )
        stepper = _make_stepper()
        result = await _step(stepper, StopTimeDiagnostics(), diagnostic_orchestrator=diag)
        assert isinstance(result, EvictingAndRestarting)
        assert result.restart.bad_node_ids == ["node-B"]
        assert isinstance(result.failed_next_state, NotifyHumans)

    @pytest.mark.asyncio
    async def test_no_bad_nodes_goes_to_notify(self) -> None:
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=[], reason="all passed"),
        )
        stepper = _make_stepper()
        result = await _step(stepper, StopTimeDiagnostics(), diagnostic_orchestrator=diag)
        assert isinstance(result, NotifyHumans)
        assert result.state_before == "StopTimeDiagnostics"



# ---------------------------------------------------------------------------
# NotifyHumans
# ---------------------------------------------------------------------------


class TestNotifyHumans:
    @pytest.mark.asyncio
    async def test_notify_returns_recovery_done(self) -> None:
        notifier = FakeNotifier()
        stepper = _make_stepper()
        result = await _step(stepper, NotifyHumans(state_before="Test"), notifier=notifier)
        assert isinstance(result, RecoveryDone)
        assert len(notifier.calls) == 1
        assert "human intervention" in notifier.calls[0][1].lower()

    @pytest.mark.asyncio
    async def test_notify_humans_with_none_notifier_does_not_crash(self) -> None:
        stepper = _make_stepper()
        result = await _step(stepper, NotifyHumans(state_before="Test"), notifier=None)
        assert isinstance(result, RecoveryDone)


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------


class TestTerminal:
    @pytest.mark.asyncio
    async def test_recovery_done_is_terminal(self) -> None:
        stepper = _make_stepper()
        result = await _step(stepper, RecoveryDone())
        assert result is None


# ---------------------------------------------------------------------------
# Global timeout (pre_dispatch)
# ---------------------------------------------------------------------------


class TestGlobalTimeout:
    @pytest.mark.asyncio
    async def test_timeout_forces_notify_humans(self) -> None:
        stepper = _make_stepper()
        old_time = _now() - timedelta(seconds=120)
        ctx = _make_ctx(
            trigger=TriggerType.HANG,
            recovery_start_time=old_time,
            timeout_seconds=60,
        )
        result = await stepper(RealtimeChecks(), ctx)
        assert isinstance(result, NotifyHumans)

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_notify_state(self) -> None:
        stepper = _make_stepper()
        old_time = _now() - timedelta(seconds=120)
        ctx = _make_ctx(
            trigger=TriggerType.HANG,
            recovery_start_time=old_time,
            timeout_seconds=60,
        )
        state = NotifyHumans(state_before="Test")
        result = await stepper(state, ctx)
        assert isinstance(result, RecoveryDone)

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_done_state(self) -> None:
        stepper = _make_stepper()
        old_time = _now() - timedelta(seconds=120)
        ctx = _make_ctx(
            trigger=TriggerType.HANG,
            recovery_start_time=old_time,
            timeout_seconds=60,
        )
        result = await stepper(RecoveryDone(), ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Full flow: restart -> fail -> Diagnostics -> E&R -> Done
# ---------------------------------------------------------------------------


class TestFullRecoveryFlow:
    @pytest.mark.asyncio
    async def test_no_fault_direct_restart_success(self) -> None:
        """RealtimeChecks (no faults) -> EvictingAndRestarting -> RestartDone -> RecoveryDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.RUNNING])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})

        restart_stepper, restart_ctx = _make_restart_stepper_and_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
        )
        stepper = _make_stepper()
        ctx = _make_ctx(
            restart_stepper=restart_stepper,
            restart_context=restart_ctx,
        )

        # Step 1: RealtimeChecks (no pre-identified) -> EvictingAndRestarting (direct restart)
        state = await stepper(RealtimeChecks(), ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)

        # Step 2: submit
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 3: poll -> MonitoringProgress
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 4: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await stepper(state, ctx)
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

        restart_stepper, restart_ctx = _make_restart_stepper_and_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            node_manager=node_manager,
            notifier=notifier,
        )
        stepper = _make_stepper()
        ctx = _make_ctx(
            restart_stepper=restart_stepper,
            restart_context=restart_ctx,
        )

        # Step 1: RealtimeChecks with pre-identified bad nodes -> EvictingAndRestarting
        state = await stepper(RealtimeChecks(pre_identified_bad_nodes=["node-X"]), ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, Evicting)
        assert state.restart.bad_node_ids == ["node-X"]
        assert isinstance(state.failed_next_state, StopTimeDiagnostics)

        # Step 2: Evicting -> mark node bad -> StoppingAndRestarting
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-X")

        # Step 3: submit
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 4: poll -> RUNNING -> MonitoringProgress
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 5: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await stepper(state, ctx)
        assert isinstance(state, RecoveryDone)

    @pytest.mark.anyio
    async def test_direct_restart_fail_escalation_full_flow(self) -> None:
        """Restart fail -> StopTimeDiagnostics -> diagnostics find bad nodes ->
        EvictingAndRestarting (notify on fail) -> RestartDone -> RecoveryDone."""
        training_job = FakeTrainingJob(status_sequence=[JobStatus.FAILED])
        mini_wandb = MiniWandb()
        mini_wandb.set_active_run_id("r")
        mini_wandb.log_step(run_id="r", step=1, metrics={"iteration": 100})
        node_manager = FakeNodeManager()

        restart_stepper, restart_ctx = _make_restart_stepper_and_context(
            training_job=training_job,
            mini_wandb=mini_wandb,
            node_manager=node_manager,
        )
        diag = FakeDiagOrchestrator(
            result=DiagnosticPipelineResult(bad_node_ids=["node-B"], reason="gpu fail"),
        )
        stepper = _make_stepper()
        ctx = _make_ctx(
            restart_stepper=restart_stepper,
            restart_context=restart_ctx,
            diagnostic_orchestrator=diag,
        )

        # Step 1: RealtimeChecks (no pre-identified) -> EvictingAndRestarting (direct restart)
        state = await stepper(RealtimeChecks(), ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)

        # Step 2: submit
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 3: poll -> FAILED -> RestartFailed -> StopTimeDiagnostics
        state = await stepper(state, ctx)
        assert isinstance(state, StopTimeDiagnostics)

        # Step 4: diagnostics find bad nodes -> EvictingAndRestarting (notify on fail)
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.failed_next_state, NotifyHumans)
        assert state.restart.bad_node_ids == ["node-B"]
        assert diag.call_count == 1

        # Switch training job to succeed for the eviction restart path
        training_job._status_sequence = [JobStatus.RUNNING]

        # Step 5: Evicting -> mark node bad -> StoppingAndRestarting
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert node_manager.is_node_bad("node-B")

        # Step 6: submit
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, StoppingAndRestarting)
        assert state.restart.submitted

        # Step 7: poll -> RUNNING -> MonitoringProgress
        state = await stepper(state, ctx)
        assert isinstance(state, EvictingAndRestarting)
        assert isinstance(state.restart, MonitoringProgress)

        # Step 8: monitoring success
        mini_wandb.log_step(run_id="r", step=2, metrics={"iteration": 200})
        state = await stepper(state, ctx)
        assert isinstance(state, RecoveryDone)

    @pytest.mark.asyncio
    async def test_notify_humans_then_done(self) -> None:
        """NotifyHumans -> RecoveryDone."""
        notifier = FakeNotifier()
        stepper = _make_stepper()
        result = await _step(stepper, NotifyHumans(state_before="Test"), notifier=notifier)
        assert isinstance(result, RecoveryDone)
