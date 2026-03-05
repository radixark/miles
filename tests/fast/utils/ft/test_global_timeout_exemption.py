"""Tests for RecoveryOrchestrator._check_global_timeout() NOTIFY/DONE exemption."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.models import RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeNotifier,
    make_fake_metric_store,
    make_fake_mini_wandb,
    FakeDiagnosticScheduler,
    FakeNodeManager,
    FakeTrainingJob,
)
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator


def _make_timed_out_orchestrator(
    phase: RecoveryPhase,
) -> RecoveryOrchestrator:
    """Create an orchestrator that has already exceeded global_timeout."""
    orch = RecoveryOrchestrator(
        trigger="crash",
        node_manager=FakeNodeManager(),
        training_job=FakeTrainingJob(),
        metric_store=make_fake_metric_store(),
        mini_wandb=make_fake_mini_wandb(),
        notifier=FakeNotifier(),
        diagnostic_scheduler=FakeDiagnosticScheduler(),
        global_timeout_seconds=0,
    )
    orch._context.phase = phase
    orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    return orch


class TestGlobalTimeoutExemption:
    def test_notify_phase_not_affected_by_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.NOTIFY)

        timed_out = orch._check_global_timeout()

        assert timed_out is False
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_done_phase_not_affected_by_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.DONE)

        timed_out = orch._check_global_timeout()

        assert timed_out is False
        assert orch.phase == RecoveryPhase.DONE

    def test_check_alerts_phase_transitions_on_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.CHECK_ALERTS)

        timed_out = orch._check_global_timeout()

        assert timed_out is True
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_monitoring_phase_transitions_on_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.MONITORING)

        timed_out = orch._check_global_timeout()

        assert timed_out is True
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_diagnosing_phase_transitions_on_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.DIAGNOSING)

        timed_out = orch._check_global_timeout()

        assert timed_out is True
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_reattempting_phase_transitions_on_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.REATTEMPTING)

        timed_out = orch._check_global_timeout()

        assert timed_out is True
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_evict_and_restart_phase_transitions_on_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.EVICT_AND_RESTART)

        timed_out = orch._check_global_timeout()

        assert timed_out is True
        assert orch.phase == RecoveryPhase.NOTIFY


class TestNotifyPhaseNoTimeoutLoop:
    """Ensure that step() from NOTIFY phase always reaches DONE, even if timed out."""

    def test_notify_step_reaches_done_despite_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.NOTIFY)

        asyncio.run(orch.step())

        assert orch.phase == RecoveryPhase.DONE
        assert orch.is_done()
