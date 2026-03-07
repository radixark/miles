from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb as MiniWandbCls
from miles.utils.ft.controller.recovery.context import RecoveryContext
from miles.utils.ft.controller.recovery.orchestrator import RecoveryOrchestrator
from miles.utils.ft.models.metric_names import CONTROLLER_RECOVERY_PHASE
from miles.utils.ft.models.fault import ActionType, Decision
from miles.utils.ft.models.recovery import RECOVERY_PHASE_TO_INT, RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticOrchestrator,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    failing_mark_node_bad,
    failing_stop_training,
    failing_submit_training,
    get_sample_value,
    inject_critical_xid,
    inject_gpu_unavailable,
    inject_nic_down,
    inject_nic_up,
    make_fake_metric_store,
    make_fake_mini_wandb,
    make_test_exporter,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_OrchestratorWithStore = tuple[
    RecoveryOrchestrator,
    FakeNodeManager,
    FakeTrainingJob,
    FakeNotifier | None,
    FakeDiagnosticOrchestrator,
    MiniPrometheus,
    MiniWandbCls,
]


def _make_orchestrator_with_store(
    trigger: str = "crash",
    status_sequence: list[JobStatus] | None = None,
    notifier: FakeNotifier | None = None,
    diagnostic_decision: Decision | None = None,
    controller_exporter: ControllerExporter | None = None,
    global_timeout_seconds: int = 1800,
    monitoring_success_iterations: int = 10,
    monitoring_timeout_seconds: int = 600,
) -> _OrchestratorWithStore:
    node_manager = FakeNodeManager()
    training_job = FakeTrainingJob(status_sequence=status_sequence)
    metric_store = make_fake_metric_store()
    mini_wandb = make_fake_mini_wandb()
    diag_orchestrator = FakeDiagnosticOrchestrator(decision=diagnostic_decision)

    orch = RecoveryOrchestrator(
        trigger=trigger,
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        diagnostic_orchestrator=diag_orchestrator,
        controller_exporter=controller_exporter,
        global_timeout_seconds=global_timeout_seconds,
        monitoring_success_iterations=monitoring_success_iterations,
        monitoring_timeout_seconds=monitoring_timeout_seconds,
    )

    return orch, node_manager, training_job, notifier, diag_orchestrator, metric_store, mini_wandb


def _make_orchestrator(
    **kwargs: object,
) -> tuple[
    RecoveryOrchestrator,
    FakeNodeManager,
    FakeTrainingJob,
    FakeNotifier | None,
    FakeDiagnosticOrchestrator,
]:
    orch, node_mgr, job, notif, diag, _, _ = _make_orchestrator_with_store(**kwargs)  # type: ignore[arg-type]
    return orch, node_mgr, job, notif, diag


# -------------------------------------------------------------------
# RecoveryPhase + RecoveryContext
# -------------------------------------------------------------------


class TestRecoveryPhase:
    def test_enum_values(self) -> None:
        assert RecoveryPhase.CHECK_ALERTS == "check_alerts"
        assert RecoveryPhase.REATTEMPTING == "reattempting"
        assert RecoveryPhase.MONITORING == "monitoring"
        assert RecoveryPhase.DIAGNOSING == "diagnosing"
        assert RecoveryPhase.EVICT_AND_RESTART == "evict_and_restart"
        assert RecoveryPhase.NOTIFY == "notify"
        assert RecoveryPhase.DONE == "done"
        assert len(RecoveryPhase) == 7

    def test_phase_to_int_mapping(self) -> None:
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.CHECK_ALERTS] == 1
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.REATTEMPTING] == 2
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.MONITORING] == 3
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.DIAGNOSING] == 4
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.EVICT_AND_RESTART] == 5
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.NOTIFY] == 6
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.DONE] == 7
        assert len(RECOVERY_PHASE_TO_INT) == len(RecoveryPhase)


class TestRecoveryContext:
    def test_defaults(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        assert ctx.trigger == "crash"
        assert ctx.phase == RecoveryPhase.CHECK_ALERTS
        assert ctx.reattempt.start_time is None
        assert ctx.reattempt.base_iteration is None
        assert ctx.global_timeout_seconds == 1800
        assert ctx.monitoring_success_iterations == 10
        assert ctx.monitoring_timeout_seconds == 600
        assert ctx.recovery_start_time.tzinfo == timezone.utc

    def test_recovery_start_time_is_utc(self) -> None:
        before = datetime.now(timezone.utc)
        ctx = RecoveryContext(trigger="hang")
        after = datetime.now(timezone.utc)
        assert before <= ctx.recovery_start_time <= after


# -------------------------------------------------------------------
# RecoveryOrchestrator: core
# -------------------------------------------------------------------


class TestRecoveryOrchestratorCore:
    def test_initial_phase(self) -> None:
        orch, *_ = _make_orchestrator()
        assert orch.phase == RecoveryPhase.CHECK_ALERTS
        assert not orch.is_done()

    def test_global_timeout_transitions_to_notify(self) -> None:
        orch, _, _, _, _ = _make_orchestrator(global_timeout_seconds=0)
        orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_step_after_done_is_noop(self) -> None:
        orch, *_ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DONE
        asyncio.run(orch.step())
        assert orch.is_done()


# -------------------------------------------------------------------
# CHECK_ALERTS phase
# -------------------------------------------------------------------


class TestCheckAlerts:
    def test_no_alerts_transitions_to_reattempting(self) -> None:
        orch, *_ = _make_orchestrator()
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.REATTEMPTING

    def test_gpu_lost_transitions_to_evict(self) -> None:
        orch, node_mgr, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_gpu_unavailable(metric_store, node_id="node-0")
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART

    def test_critical_xid_transitions_to_evict(self) -> None:
        orch, node_mgr, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_critical_xid(metric_store, node_id="node-1")
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART


# -------------------------------------------------------------------
# REATTEMPTING phase
# -------------------------------------------------------------------


class TestReattempting:
    def test_submit_then_running_transitions_to_monitoring(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        asyncio.run(orch.step())
        assert orch._context.reattempt.submitted
        assert training_job._stopped
        assert training_job._submitted

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.MONITORING

    def test_submit_then_failed_transitions_to_diagnosing(self) -> None:
        orch, _, _, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.FAILED],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        asyncio.run(orch.step())  # submit
        asyncio.run(orch.step())  # poll -> FAILED
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_pending_timeout_transitions_to_notify(self) -> None:
        orch, _, _, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.PENDING, JobStatus.PENDING],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        asyncio.run(orch.step())  # submit
        orch._context.reattempt.submit_time = datetime.now(timezone.utc) - timedelta(seconds=301)
        asyncio.run(orch.step())  # poll -> PENDING timeout
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_stop_training_exception_continues(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.STOPPED],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        training_job.stop_training = failing_stop_training

        asyncio.run(orch.step())
        assert orch._context.reattempt.submitted
        assert training_job._submitted


# -------------------------------------------------------------------
# MONITORING phase
# -------------------------------------------------------------------


class TestMonitoring:
    def test_success_after_enough_iterations(self) -> None:
        orch, _, _, _, _, _, mini_wandb = _make_orchestrator_with_store(
            status_sequence=[JobStatus.RUNNING],
            monitoring_success_iterations=3,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt.start_time = datetime.now(timezone.utc)
        orch._context.reattempt.base_iteration = 0

        for i in range(1, 4):
            mini_wandb.log_step(
                run_id="test-run", step=i,
                metrics={"iteration": float(i)},
            )

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE

    def test_failed_at_first_iteration_goes_to_diagnosing(self) -> None:
        orch, *_ = _make_orchestrator(
            status_sequence=[JobStatus.FAILED],
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt.start_time = datetime.now(timezone.utc)
        orch._context.reattempt.base_iteration = 0

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_failed_after_some_iterations_goes_to_diagnosing(self) -> None:
        orch, _, _, _, _, _, mini_wandb = _make_orchestrator_with_store(
            status_sequence=[JobStatus.FAILED],
            monitoring_success_iterations=10,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt.start_time = datetime.now(timezone.utc)
        orch._context.reattempt.base_iteration = 0

        for i in range(1, 6):
            mini_wandb.log_step(
                run_id="test-run", step=i,
                metrics={"iteration": float(i)},
            )

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_monitoring_timeout_goes_to_diagnosing(self) -> None:
        orch, *_ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
            monitoring_timeout_seconds=60,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt.start_time = datetime.now(timezone.utc) - timedelta(seconds=61)
        orch._context.reattempt.base_iteration = 0

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_running_no_progress_waits(self) -> None:
        orch, *_ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
            monitoring_timeout_seconds=600,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt.start_time = datetime.now(timezone.utc)
        orch._context.reattempt.base_iteration = 0

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.MONITORING


# -------------------------------------------------------------------
# DIAGNOSING phase
# -------------------------------------------------------------------


class TestDiagnosing:
    def test_diagnostic_finds_bad_node(self) -> None:
        bad_decision = Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-2"],
            reason="GPU failed diagnostic",
        )
        orch, _, _, _, diag = _make_orchestrator(diagnostic_decision=bad_decision)
        orch._context.phase = RecoveryPhase.DIAGNOSING

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART
        assert diag.call_count == 1
        assert orch._context.bad_node_ids == ["node-2"]

    def test_diagnostic_all_passed_goes_to_notify(self) -> None:
        orch, _, _, _, diag = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DIAGNOSING

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY
        assert diag.call_count == 1


# -------------------------------------------------------------------
# EVICT_AND_RESTART phase
# -------------------------------------------------------------------


class TestEvictAndRestart:
    def test_normal_evict_and_restart(self) -> None:
        orch, node_mgr, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-0"]

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert node_mgr.is_node_bad("node-0")
        assert training_job._submitted

    def test_evict_passes_excluded_node_ids_to_submit(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-a", "node-b"]

        asyncio.run(orch.step())
        assert training_job._last_excluded_node_ids == ["node-a", "node-b"]

    def test_mark_node_bad_fails_goes_to_notify(self) -> None:
        orch, node_mgr, _, notifier, _ = _make_orchestrator(
            notifier=FakeNotifier(),
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-0"]

        node_mgr.mark_node_bad = failing_mark_node_bad

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_empty_bad_node_ids_skips_to_notify(self) -> None:
        orch, node_mgr, training_job, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = []

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY
        assert not node_mgr._bad_nodes
        assert not training_job._submitted

    def test_submit_training_fails_goes_to_notify(self) -> None:
        orch, _, training_job, notifier, _ = _make_orchestrator(
            notifier=FakeNotifier(),
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-0"]

        training_job.submit_training = failing_submit_training

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY


# -------------------------------------------------------------------
# NOTIFY phase
# -------------------------------------------------------------------


class TestNotify:
    def test_notify_sends_and_transitions_to_done(self) -> None:
        notifier = FakeNotifier()
        orch, _, _, _, _ = _make_orchestrator(notifier=notifier)
        orch._context.phase = RecoveryPhase.NOTIFY

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert len(notifier.calls) == 1
        assert notifier.calls[0][2] == "critical"

    def test_notify_without_notifier(self) -> None:
        orch, *_ = _make_orchestrator(notifier=None)
        orch._context.phase = RecoveryPhase.NOTIFY

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE


# -------------------------------------------------------------------
# Exporter integration
# -------------------------------------------------------------------


class TestCheckAlertsNicDown:
    def test_majority_nic_down_transitions_to_evict(self) -> None:
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_down(metric_store, node_id="node-0", device="ib1")
        inject_nic_down(metric_store, node_id="node-0", device="ib2")
        inject_nic_up(metric_store, node_id="node-0", device="ib3")
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART

    def test_minority_nic_down_proceeds_to_reattempting(self) -> None:
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_up(metric_store, node_id="node-0", device="ib1")
        inject_nic_up(metric_store, node_id="node-0", device="ib2")
        inject_nic_up(metric_store, node_id="node-0", device="ib3")
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.REATTEMPTING


class TestCheckAlertsEphemeral:
    def test_ephemeral_only_nic_flapping_goes_to_reattempting(self) -> None:
        """NIC flapping without hardware faults should go to REATTEMPTING, not EVICT."""
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        # Inject 2 NIC-down samples for same device → triggers check_nic_down_in_window (threshold=2)
        # but keep majority of NICs up → does NOT trigger _check_majority_nic_down
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_up(metric_store, node_id="node-0", device="ib1")
        inject_nic_up(metric_store, node_id="node-0", device="ib2")
        inject_nic_up(metric_store, node_id="node-0", device="ib3")

        asyncio.run(orch.step())

        assert orch.phase == RecoveryPhase.REATTEMPTING
        assert orch.bad_node_ids == []

    def test_hardware_plus_ephemeral_goes_to_evict_with_all_nodes(self) -> None:
        """Hardware fault + NIC flapping should evict all affected nodes."""
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        # node-1: GPU fault (non-ephemeral)
        inject_gpu_unavailable(metric_store, node_id="node-1")
        # node-0: NIC flapping only (ephemeral)
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_down(metric_store, node_id="node-0", device="ib0")
        inject_nic_up(metric_store, node_id="node-0", device="ib1")
        inject_nic_up(metric_store, node_id="node-0", device="ib2")
        inject_nic_up(metric_store, node_id="node-0", device="ib3")

        asyncio.run(orch.step())

        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART
        assert "node-0" in orch.bad_node_ids
        assert "node-1" in orch.bad_node_ids


class TestCheckAlertsXidCodes:
    def test_zero_non_auto_recoverable_counter_ignored(self) -> None:
        """A zero non-auto-recoverable counter does not trigger eviction."""
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        from miles.utils.ft.models.metric_names import XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL
        from miles.utils.ft.models.metrics import CounterSample
        metric_store.ingest_samples(target_id="node-0", samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=0.0),
        ])
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.REATTEMPTING


class TestReattemptingSubmitFailure:
    def test_submit_failure_transitions_to_notify(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        training_job.submit_training = failing_submit_training

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY


class TestNotifyWithBrokenNotifier:
    def test_notifier_exception_still_transitions_to_done(self) -> None:
        notifier = FakeNotifier()

        async def broken_send(title: str, content: str, severity: str) -> None:
            raise RuntimeError("notification service unreachable")
        notifier.send = broken_send

        orch, _, _, _, _ = _make_orchestrator(notifier=notifier)
        orch._context.phase = RecoveryPhase.NOTIFY

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE


class TestEvictStopTrainingFailure:
    def test_stop_failure_continues_to_submit(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.STOPPED],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-0"]

        training_job.stop_training = failing_stop_training

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert training_job._submitted


class TestEvictMultipleBadNodes:
    def test_multiple_bad_nodes_all_marked(self) -> None:
        orch, node_mgr, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._context.bad_node_ids = ["node-0", "node-1", "node-2"]

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert node_mgr.is_node_bad("node-0")
        assert node_mgr.is_node_bad("node-1")
        assert node_mgr.is_node_bad("node-2")
        assert training_job._submitted


class TestPhaseBeforeNotify:
    def test_global_timeout_captures_previous_phase(self) -> None:
        orch, _, _, _, _ = _make_orchestrator(global_timeout_seconds=0)
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=1)

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY
        assert orch._context.phase_before_notify == RecoveryPhase.MONITORING

    def test_diagnosing_all_passed_captures_previous_phase(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DIAGNOSING

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY
        assert orch._context.phase_before_notify == RecoveryPhase.DIAGNOSING


class TestExporterIntegration:
    def test_exporter_updated_on_transition(self) -> None:
        registry, exporter = make_test_exporter()
        orch, *_ = _make_orchestrator(controller_exporter=exporter)

        asyncio.run(orch.step())

        phase_value = get_sample_value(registry, CONTROLLER_RECOVERY_PHASE)
        assert phase_value == RECOVERY_PHASE_TO_INT[RecoveryPhase.REATTEMPTING]


class TestStepWhenDoneIsNoop:
    def test_step_after_done_does_not_change_state(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DONE
        history_before = list(orch.phase_history)

        asyncio.run(orch.step())

        assert orch.phase == RecoveryPhase.DONE
        assert orch.phase_history == history_before


class TestDispatchPhaseUnknown:
    def test_unknown_phase_returns_none_no_transition(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DONE
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE


class TestAddBadNodesEscalation:
    def test_add_bad_nodes_during_reattempting_escalates(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.REATTEMPTING
        orch._context.bad_node_ids = ["node-0"]

        orch.add_bad_nodes(["node-1"])

        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART
        assert "node-0" in orch.bad_node_ids
        assert "node-1" in orch.bad_node_ids

    def test_add_bad_nodes_during_monitoring_escalates(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.bad_node_ids = []

        orch.add_bad_nodes(["node-2"])

        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART
        assert "node-2" in orch.bad_node_ids

    def test_add_duplicate_bad_nodes_is_noop(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.REATTEMPTING
        orch._context.bad_node_ids = ["node-0"]
        phase_before = orch.phase

        orch.add_bad_nodes(["node-0"])

        assert orch.phase == phase_before
        assert orch.bad_node_ids == ["node-0"]

    def test_add_bad_nodes_during_diagnosing_does_not_escalate(self) -> None:
        orch, _, _, _, _ = _make_orchestrator()
        orch._context.phase = RecoveryPhase.DIAGNOSING
        orch._context.bad_node_ids = []

        orch.add_bad_nodes(["node-3"])

        assert orch.phase == RecoveryPhase.DIAGNOSING
        assert "node-3" in orch.bad_node_ids


# -------------------------------------------------------------------
# Global timeout exemption
# -------------------------------------------------------------------


def _make_timed_out_orchestrator(
    phase: RecoveryPhase,
) -> RecoveryOrchestrator:
    """Create an orchestrator that has already exceeded global_timeout."""
    orch, *_ = _make_orchestrator_with_store(
        global_timeout_seconds=0,
        notifier=FakeNotifier(),
    )
    orch._context.phase = phase
    orch._context.recovery_start_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    return orch


class TestGlobalTimeoutExemption:
    @pytest.mark.parametrize("phase,expected_timed_out,expected_phase", [
        (RecoveryPhase.NOTIFY, False, RecoveryPhase.NOTIFY),
        (RecoveryPhase.DONE, False, RecoveryPhase.DONE),
        (RecoveryPhase.CHECK_ALERTS, True, RecoveryPhase.NOTIFY),
        (RecoveryPhase.MONITORING, True, RecoveryPhase.NOTIFY),
        (RecoveryPhase.DIAGNOSING, True, RecoveryPhase.NOTIFY),
        (RecoveryPhase.REATTEMPTING, True, RecoveryPhase.NOTIFY),
        (RecoveryPhase.EVICT_AND_RESTART, True, RecoveryPhase.NOTIFY),
    ])
    def test_global_timeout_per_phase(
        self,
        phase: RecoveryPhase,
        expected_timed_out: bool,
        expected_phase: RecoveryPhase,
    ) -> None:
        orch = _make_timed_out_orchestrator(phase)
        timed_out = orch._check_global_timeout()
        assert timed_out is expected_timed_out
        assert orch.phase == expected_phase


class TestNotifyPhaseNoTimeoutLoop:
    """Ensure that step() from NOTIFY phase always reaches DONE, even if timed out."""

    def test_notify_step_reaches_done_despite_timeout(self) -> None:
        orch = _make_timed_out_orchestrator(RecoveryPhase.NOTIFY)

        asyncio.run(orch.step())

        assert orch.phase == RecoveryPhase.DONE
        assert orch.is_done()


