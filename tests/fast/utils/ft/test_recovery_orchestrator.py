from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from prometheus_client import CollectorRegistry

from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.mini_wandb import MiniWandb as MiniWandbCls
from miles.utils.ft.controller.recovery_orchestrator import (
    RecoveryContext,
    RecoveryOrchestrator,
)
from miles.utils.ft.models import (
    ActionType,
    Decision,
    RECOVERY_PHASE_TO_INT,
    RecoveryPhase,
)
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticScheduler,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    get_sample_value,
    inject_critical_xid,
    inject_disk_fault,
    inject_gpu_unavailable,
    inject_nic_down,
    inject_nic_up,
    make_fake_metric_store,
    make_fake_mini_wandb,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_OrchestratorWithStore = tuple[
    RecoveryOrchestrator,
    FakeNodeManager,
    FakeTrainingJob,
    FakeNotifier | None,
    FakeDiagnosticScheduler,
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
    diag_scheduler = FakeDiagnosticScheduler(decision=diagnostic_decision)

    orch = RecoveryOrchestrator(
        trigger=trigger,
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        diagnostic_scheduler=diag_scheduler,
        controller_exporter=controller_exporter,
        global_timeout_seconds=global_timeout_seconds,
        monitoring_success_iterations=monitoring_success_iterations,
        monitoring_timeout_seconds=monitoring_timeout_seconds,
    )
    return orch, node_manager, training_job, notifier, diag_scheduler, metric_store, mini_wandb


def _make_orchestrator(
    **kwargs: object,
) -> tuple[
    RecoveryOrchestrator,
    FakeNodeManager,
    FakeTrainingJob,
    FakeNotifier | None,
    FakeDiagnosticScheduler,
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
        assert ctx.reattempt_start_time is None
        assert ctx.reattempt_base_iteration is None
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
        inject_critical_xid(metric_store, node_id="node-1", xid_code=48)
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
        assert orch._reattempt_submitted
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
        orch._reattempt_submit_time = datetime.now(timezone.utc) - timedelta(seconds=301)
        asyncio.run(orch.step())  # poll -> PENDING timeout
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_stop_training_exception_continues(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")
        training_job.stop_training = failing_stop

        asyncio.run(orch.step())
        assert orch._reattempt_submitted
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
        orch._context.reattempt_start_time = datetime.now(timezone.utc)
        orch._context.reattempt_base_iteration = 0

        for i in range(1, 4):
            mini_wandb.log_step(
                run_id="test-run", rank=0, step=i,
                metrics={"iteration": float(i)},
            )

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE

    def test_failed_at_first_iteration_goes_to_diagnosing(self) -> None:
        orch, *_ = _make_orchestrator(
            status_sequence=[JobStatus.FAILED],
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt_start_time = datetime.now(timezone.utc)
        orch._context.reattempt_base_iteration = 0

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_failed_after_some_iterations_goes_to_diagnosing(self) -> None:
        orch, _, _, _, _, _, mini_wandb = _make_orchestrator_with_store(
            status_sequence=[JobStatus.FAILED],
            monitoring_success_iterations=10,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt_start_time = datetime.now(timezone.utc)
        orch._context.reattempt_base_iteration = 0

        for i in range(1, 6):
            mini_wandb.log_step(
                run_id="test-run", rank=0, step=i,
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
        orch._context.reattempt_start_time = datetime.now(timezone.utc) - timedelta(seconds=61)
        orch._context.reattempt_base_iteration = 0

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DIAGNOSING

    def test_running_no_progress_waits(self) -> None:
        orch, *_ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
            monitoring_timeout_seconds=600,
        )
        orch._context.phase = RecoveryPhase.MONITORING
        orch._context.reattempt_start_time = datetime.now(timezone.utc)
        orch._context.reattempt_base_iteration = 0

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
        assert orch._bad_node_ids == ["node-2"]

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
        orch._bad_node_ids = ["node-0"]

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert node_mgr.is_node_bad("node-0")
        assert training_job._submitted

    def test_mark_node_bad_fails_goes_to_notify(self) -> None:
        orch, node_mgr, _, notifier, _ = _make_orchestrator(
            notifier=FakeNotifier(),
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._bad_node_ids = ["node-0"]

        async def failing_mark(node_id: str, reason: str = "") -> None:
            raise RuntimeError("K8s API unreachable")
        node_mgr.mark_node_bad = failing_mark

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.NOTIFY

    def test_submit_training_fails_goes_to_notify(self) -> None:
        orch, _, training_job, notifier, _ = _make_orchestrator(
            notifier=FakeNotifier(),
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._bad_node_ids = ["node-0"]

        async def failing_submit() -> str:
            raise RuntimeError("submit failed")
        training_job.submit_training = failing_submit

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


class TestCheckAlertsDiskFault:
    def test_disk_fault_transitions_to_evict(self) -> None:
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_disk_fault(metric_store, node_id="node-0", available_bytes=0.0)
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.EVICT_AND_RESTART


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


class TestCheckAlertsXidCodes:
    def test_non_critical_xid_ignored(self) -> None:
        orch, _, _, _, _, metric_store, _ = _make_orchestrator_with_store()
        inject_critical_xid(metric_store, node_id="node-0", xid_code=13)
        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.REATTEMPTING


class TestReattemptingSubmitFailure:
    def test_submit_failure_transitions_to_notify(self) -> None:
        orch, _, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.REATTEMPTING

        async def failing_submit() -> str:
            raise RuntimeError("submit failed")
        training_job.submit_training = failing_submit

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
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._bad_node_ids = ["node-0"]

        async def failing_stop(timeout_seconds: int = 300) -> None:
            raise RuntimeError("stop failed")
        training_job.stop_training = failing_stop

        asyncio.run(orch.step())
        assert orch.phase == RecoveryPhase.DONE
        assert training_job._submitted


class TestEvictMultipleBadNodes:
    def test_multiple_bad_nodes_all_marked(self) -> None:
        orch, node_mgr, training_job, _, _ = _make_orchestrator(
            status_sequence=[JobStatus.RUNNING],
        )
        orch._context.phase = RecoveryPhase.EVICT_AND_RESTART
        orch._bad_node_ids = ["node-0", "node-1", "node-2"]

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
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        orch, *_ = _make_orchestrator(controller_exporter=exporter)

        asyncio.run(orch.step())

        phase_value = get_sample_value(registry, "ft_controller_recovery_phase")
        assert phase_value == RECOVERY_PHASE_TO_INT[RecoveryPhase.REATTEMPTING]
