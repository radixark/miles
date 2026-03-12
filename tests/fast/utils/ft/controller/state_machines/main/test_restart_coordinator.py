"""Tests for restart_coordinator: trigger_main_job_restart and resolve_main_job_restart.

Previously the restart logic (stop_and_submit dispatch, job status polling,
requestor-state restoration) lived directly inside NormalHandler and
RestartingMainJobHandler, making it impossible to unit-test the three
restart outcomes (SUCCEEDED / FAILED / TIMEOUT) without running the
full state machine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from prometheus_client import CollectorRegistry
from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob, FakeNodeManager, FakeNotifier, make_failing_main_job
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeDiagnosticOrchestrator

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import (
    MainContext,
    NormalSt,
    RestartingMainJobSt,
)
from miles.utils.ft.controller.state_machines.main.restart_coordinator import (
    resolve_main_job_restart,
    trigger_main_job_restart,
)
from miles.utils.ft.controller.state_machines.recovery.models import EvictingAndRestartingSt, StopTimeDiagnosticsSt
from miles.utils.ft.controller.state_machines.restart.models import (
    ExternalExecutionResult,
    ExternalRestartingMainJobSt,
    MonitoringIterationProgressConfig,
)
from miles.utils.ft.controller.state_machines.subsystem.models import DetectingAnomalySt, RecoveringSt
from miles.utils.ft.controller.subsystem_hub.config import RestartMode, SubsystemConfig, SubsystemRuntime, SubsystemSpec
from miles.utils.ft.controller.types import MetricStore, SharedDeps, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter


def _make_shared_deps(
    *,
    main_job: FakeMainJob | None = None,
    notifier: FakeNotifier | None = None,
    recovery_timeout_seconds: int = 3600,
) -> SharedDeps:
    resolved_main_job = main_job or FakeMainJob()
    return SharedDeps(
        main_job=resolved_main_job,
        subsystem_specs={"training": _make_dummy_spec()},
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        notifier=notifier,
        node_manager=FakeNodeManager(),
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        detector_crash_tracker=SlidingWindowCounter(window_minutes=30, max_count=3),
        recovery_timeout_seconds=recovery_timeout_seconds,
        max_simultaneous_bad_nodes=2,
        on_main_job_new_run=None,
        rank_pids_provider=None,
        controller_exporter=ControllerExporter(registry=CollectorRegistry()),
        on_recovery_duration=None,
        registration_grace_ticks=0,
    )


def _make_dummy_spec() -> SubsystemSpec:
    from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

    class _StubActuator:
        async def start(self) -> str:
            return "stub"

        async def stop(self, timeout_seconds: int = 300) -> None:
            pass

        async def get_status(self) -> JobStatus:
            return JobStatus.RUNNING

    config = SubsystemConfig(
        detectors=[],
        monitoring_config=MonitoringIterationProgressConfig(),
        restart_mode=RestartMode.MAIN_JOB,
    )
    runtime = SubsystemRuntime(
        actuator=_StubActuator(),  # type: ignore[arg-type]
        cooldown=SlidingWindowThrottle(window_minutes=30, max_count=3),
        get_active_node_ids=lambda: frozenset({"node-0"}),
    )
    return SubsystemSpec(config=config, runtime=runtime)


def _make_context(
    *,
    shared: SharedDeps | None = None,
    job_status: JobStatus = JobStatus.RUNNING,
) -> MainContext:
    return MainContext(
        shared=shared or _make_shared_deps(),
        tick_count=10,
        run_start_tick=0,
        job_status=job_status,
        node_metadata={},
    )


def _make_requestor_state() -> RecoveringSt:
    return RecoveringSt(
        recovery=EvictingAndRestartingSt(
            restart=ExternalRestartingMainJobSt(external_execution_result=None),
            failed_next_state=StopTimeDiagnosticsSt(),
        ),
        trigger=TriggerType.CRASH,
        recovery_start_time=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# trigger_main_job_restart
# ---------------------------------------------------------------------------


class TestTriggerMainJobRestart:
    @pytest.mark.anyio
    async def test_returns_none_when_no_requestor(self) -> None:
        state = NormalSt(subsystems={"training": DetectingAnomalySt()})
        context = _make_context()

        result = await trigger_main_job_restart(state=state, context=context)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_restarting_main_job_on_success(self) -> None:
        """When stop_and_submit succeeds, returns RestartingMainJobSt."""
        state = NormalSt(subsystems={"training": _make_requestor_state()})
        context = _make_context()

        result = await trigger_main_job_restart(state=state, context=context)

        assert isinstance(result, RestartingMainJobSt)
        assert result.requestor_name == "training"

    @pytest.mark.anyio
    async def test_returns_normal_with_failed_result_on_submit_failure(self) -> None:
        """When stop_and_submit fails, returns NormalSt with FAILED result
        patched into the requestor's state."""
        main_job = make_failing_main_job(fail_submit=True)
        notifier = FakeNotifier()
        shared = _make_shared_deps(main_job=main_job, notifier=notifier)
        state = NormalSt(subsystems={"training": _make_requestor_state()})
        context = _make_context(shared=shared)

        result = await trigger_main_job_restart(state=state, context=context)

        assert isinstance(result, NormalSt)
        training = result.subsystems["training"]
        assert isinstance(training, RecoveringSt)
        assert isinstance(training.recovery, EvictingAndRestartingSt)
        assert isinstance(training.recovery.restart, ExternalRestartingMainJobSt)
        assert training.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED
        assert len(notifier.calls) == 1
        assert "Main job restart failed" in notifier.calls[0][0]


# ---------------------------------------------------------------------------
# resolve_main_job_restart
# ---------------------------------------------------------------------------


class TestResolveMainJobRestart:
    @pytest.mark.anyio
    async def test_running_returns_normal_with_succeeded(self) -> None:
        """When the restarted main job reaches RUNNING, all subsystems are
        reset and the requestor receives SUCCEEDED."""
        main_job = FakeMainJob(status_sequence=[JobStatus.RUNNING])
        shared = _make_shared_deps(main_job=main_job)
        context = _make_context(shared=shared)
        state = RestartingMainJobSt(
            requestor_name="training",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=_make_requestor_state(),
        )

        result = await resolve_main_job_restart(state=state, context=context)

        assert isinstance(result, NormalSt)
        training = result.subsystems["training"]
        assert isinstance(training, RecoveringSt)
        assert isinstance(training.recovery.restart, ExternalRestartingMainJobSt)
        assert training.recovery.restart.external_execution_result == ExternalExecutionResult.SUCCEEDED

    @pytest.mark.anyio
    async def test_failed_returns_normal_with_failed(self) -> None:
        """When the restarted main job immediately fails, the requestor
        receives FAILED so recovery can escalate."""
        main_job = FakeMainJob(status_sequence=[JobStatus.FAILED])
        shared = _make_shared_deps(main_job=main_job)
        context = _make_context(shared=shared)
        state = RestartingMainJobSt(
            requestor_name="training",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=_make_requestor_state(),
        )

        result = await resolve_main_job_restart(state=state, context=context)

        assert isinstance(result, NormalSt)
        training = result.subsystems["training"]
        assert isinstance(training, RecoveringSt)
        assert isinstance(training.recovery.restart, ExternalRestartingMainJobSt)
        assert training.recovery.restart.external_execution_result == ExternalExecutionResult.FAILED

    @pytest.mark.anyio
    async def test_timeout_returns_normal_with_timeout(self) -> None:
        """When the restart exceeds recovery_timeout_seconds, the requestor
        receives TIMEOUT."""
        main_job = FakeMainJob(status_sequence=[JobStatus.STOPPED])
        shared = _make_shared_deps(main_job=main_job, recovery_timeout_seconds=60)
        context = _make_context(shared=shared)
        state = RestartingMainJobSt(
            requestor_name="training",
            start_time=datetime.now(timezone.utc) - timedelta(seconds=120),
            requestor_frozen_state=_make_requestor_state(),
        )

        result = await resolve_main_job_restart(state=state, context=context)

        assert isinstance(result, NormalSt)
        training = result.subsystems["training"]
        assert isinstance(training, RecoveringSt)
        assert isinstance(training.recovery.restart, ExternalRestartingMainJobSt)
        assert training.recovery.restart.external_execution_result == ExternalExecutionResult.TIMEOUT

    @pytest.mark.anyio
    async def test_pending_returns_none(self) -> None:
        """While the job is not yet running/failed and timeout not exceeded,
        returns None (no transition yet)."""
        main_job = FakeMainJob(status_sequence=[JobStatus.STOPPED])
        shared = _make_shared_deps(main_job=main_job, recovery_timeout_seconds=3600)
        context = _make_context(shared=shared)
        state = RestartingMainJobSt(
            requestor_name="training",
            start_time=datetime.now(timezone.utc),
            requestor_frozen_state=_make_requestor_state(),
        )

        result = await resolve_main_job_restart(state=state, context=context)

        assert result is None
