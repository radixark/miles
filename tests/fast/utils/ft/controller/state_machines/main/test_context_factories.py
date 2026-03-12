"""Unit tests for context_factories.py (P0 item 5)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.adapters.types import JobStatus, SubsystemActuatorProtocol
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.context_factories import (
    _build_recovery_context,
    _should_run_detectors,
    build_subsystem_context,
)
from miles.utils.ft.controller.state_machines.main.models import MainContext
from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig
from miles.utils.ft.controller.subsystem_hub.config import RestartMode, SubsystemConfig, SubsystemRuntime, SubsystemSpec
from miles.utils.ft.controller.types import MetricStore, SharedDeps, TriggerType
from miles.utils.ft.utils.sliding_window import SlidingWindowCounter

from tests.fast.utils.ft.utils.controller_fakes import FakeMainJob, FakeNodeManager, FakeNotifier
from tests.fast.utils.ft.utils.diagnostic_fakes import FakeDiagnosticOrchestrator


def _make_main_context(
    *,
    tick_count: int = 10,
    run_start_tick: int = 0,
    registration_grace_ticks: int = 5,
    job_status: JobStatus = JobStatus.RUNNING,
) -> MainContext:
    shared = SharedDeps(
        main_job=FakeMainJob(),
        subsystem_specs={},
        metric_store=MetricStore(
            time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
            mini_wandb=MiniWandb(),
        ),
        notifier=FakeNotifier(),
        node_manager=FakeNodeManager(),
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        detector_crash_tracker=SlidingWindowCounter(window_seconds=300, threshold=5),
        recovery_timeout_seconds=600,
        max_simultaneous_bad_nodes=2,
        on_main_job_new_run=None,
        rank_pids_provider=None,
        controller_exporter=None,
        on_recovery_duration=None,
        registration_grace_ticks=registration_grace_ticks,
    )
    return MainContext(
        shared=shared,
        tick_count=tick_count,
        run_start_tick=run_start_tick,
        job_status=job_status,
        node_metadata={},
    )


class TestShouldRunDetectors:
    def test_returns_false_when_active_node_ids_empty(self) -> None:
        ctx = _make_main_context(tick_count=10, registration_grace_ticks=5)
        assert _should_run_detectors(active_node_ids=frozenset(), context=ctx) is False

    def test_returns_false_during_registration_grace_period(self) -> None:
        ctx = _make_main_context(tick_count=3, registration_grace_ticks=5)
        assert _should_run_detectors(active_node_ids=frozenset({"node-0"}), context=ctx) is False

    def test_returns_false_at_exact_grace_tick_boundary(self) -> None:
        ctx = _make_main_context(tick_count=5, registration_grace_ticks=5)
        assert _should_run_detectors(active_node_ids=frozenset({"node-0"}), context=ctx) is False

    def test_returns_true_after_grace_period(self) -> None:
        ctx = _make_main_context(tick_count=6, registration_grace_ticks=5)
        assert _should_run_detectors(active_node_ids=frozenset({"node-0"}), context=ctx) is True

    def test_returns_true_with_nodes_and_no_grace(self) -> None:
        ctx = _make_main_context(tick_count=1, registration_grace_ticks=0)
        assert _should_run_detectors(active_node_ids=frozenset({"node-0", "node-1"}), context=ctx) is True

    def test_grace_period_resets_on_main_job_new_run(self) -> None:
        """After a new run starts at tick 100, grace period should apply
        relative to the run start tick, not global tick_count. Previously
        the grace was only effective at process startup."""
        ctx = _make_main_context(
            tick_count=102,
            run_start_tick=100,
            registration_grace_ticks=5,
        )
        assert _should_run_detectors(active_node_ids=frozenset({"node-0"}), context=ctx) is False

    def test_detectors_run_after_grace_period_since_new_run(self) -> None:
        """Once enough ticks pass after the new run's start tick,
        detectors should run again."""
        ctx = _make_main_context(
            tick_count=106,
            run_start_tick=100,
            registration_grace_ticks=5,
        )
        assert _should_run_detectors(active_node_ids=frozenset({"node-0"}), context=ctx) is True


class TestBuildSubsystemContext:
    def test_per_subsystem_cooldown_isolation(self) -> None:
        """Cooldown was shared across all subsystems, so a rollout recovery
        counted toward training's cooldown limit and vice versa. Now each
        subsystem gets its own cooldown from SubsystemConfig."""
        from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

        ctx = _make_main_context(tick_count=10)
        cooldown_a = SlidingWindowThrottle(window_minutes=30, max_count=3)
        cooldown_b = SlidingWindowThrottle(window_minutes=30, max_count=3)
        spec_a = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(
                actuator=MagicMock(),
                cooldown=cooldown_a,
                get_active_node_ids=lambda: frozenset({"node-0"}),
            ),
        )
        spec_b = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(
                actuator=MagicMock(),
                cooldown=cooldown_b,
                get_active_node_ids=lambda: frozenset({"node-1"}),
            ),
        )

        result_a = build_subsystem_context(
            spec=spec_a, context=ctx,
            recovery_stepper=AsyncMock(), restart_stepper=AsyncMock(),
        )
        result_b = build_subsystem_context(
            spec=spec_b, context=ctx,
            recovery_stepper=AsyncMock(), restart_stepper=AsyncMock(),
        )

        assert result_a.cooldown is cooldown_a
        assert result_b.cooldown is cooldown_b
        assert result_a.cooldown is not result_b.cooldown

    def test_wires_all_fields_correctly(self) -> None:
        ctx = _make_main_context(tick_count=10)
        actuator = MagicMock()
        monitoring_config = MonitoringIterationProgressConfig()
        spec = SubsystemSpec(
            config=SubsystemConfig(
                detectors=[],
                monitoring_config=monitoring_config,
            ),
            runtime=SubsystemRuntime(
                actuator=actuator,
                get_active_node_ids=lambda: frozenset({"node-0"}),
            ),
        )
        recovery_stepper = AsyncMock()
        restart_stepper = AsyncMock()

        result = build_subsystem_context(
            spec=spec,
            context=ctx,
            recovery_stepper=recovery_stepper,
            restart_stepper=restart_stepper,
        )

        assert result.job_status == ctx.job_status
        assert result.tick_count == ctx.tick_count
        assert result.should_run_detectors is True
        assert result.detector_context is not None
        assert result.detector_context.active_node_ids == frozenset({"node-0"})
        assert result.notifier is ctx.shared.notifier
        assert result.detectors == []
        assert result.cooldown is spec.runtime.cooldown
        assert result.recovery_stepper is recovery_stepper
        assert result.max_simultaneous_bad_nodes == ctx.shared.max_simultaneous_bad_nodes
        assert result.monitoring_config is monitoring_config

    def test_detector_context_is_none_when_detectors_should_not_run(self) -> None:
        ctx = _make_main_context(tick_count=1, registration_grace_ticks=5)
        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(
                actuator=MagicMock(),
                get_active_node_ids=lambda: frozenset({"node-0"}),
            ),
        )

        result = build_subsystem_context(
            spec=spec,
            context=ctx,
            recovery_stepper=AsyncMock(),
            restart_stepper=AsyncMock(),
        )

        assert result.should_run_detectors is False
        assert result.detector_context is None


class TestBuildRecoveryContext:
    def test_correctly_constructs_recovery_context(self) -> None:
        ctx = _make_main_context()
        actuator = AsyncMock(spec=SubsystemActuatorProtocol)
        monitoring_config = MonitoringIterationProgressConfig()
        spec = SubsystemSpec(
            config=SubsystemConfig(
                monitoring_config=monitoring_config,
            ),
            runtime=SubsystemRuntime(
                actuator=actuator,
            ),
        )
        restart_stepper = AsyncMock()
        now = datetime.now(timezone.utc)

        result = _build_recovery_context(
            spec=spec,
            context=ctx,
            trigger=TriggerType.CRASH,
            recovery_start_time=now,
            restart_stepper=restart_stepper,
        )

        assert result.trigger == TriggerType.CRASH
        assert result.recovery_start_time == now
        assert result.restart_stepper is restart_stepper
        assert result.notifier is ctx.shared.notifier
        assert result.timeout_seconds == ctx.shared.recovery_timeout_seconds

        assert result.restart_context.node_manager is ctx.shared.node_manager
        assert result.restart_context.main_job is ctx.shared.main_job
        assert result.restart_context.actuator is actuator
        assert result.restart_context.monitoring_config is monitoring_config
        assert result.restart_context.is_main_job_restart is False
        assert result.restart_context.on_new_run is None

    def test_node_metadata_propagated_to_restart_context(self) -> None:
        """node_metadata was not passed from MainContext to RestartContext,
        so EvictingHandler always received empty metadata and mark_node_bad
        used Ray node_id instead of K8s node name for K8s label operations."""
        metadata = {"ray-uuid-abc": {"k8s_node_name": "gke-node-01"}}
        shared = SharedDeps(
            main_job=FakeMainJob(),
            subsystem_specs={},
            metric_store=MetricStore(
                time_series_store=MiniPrometheus(config=MiniPrometheusConfig()),
                mini_wandb=MiniWandb(),
            ),
            notifier=FakeNotifier(),
            node_manager=FakeNodeManager(),
            diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
            detector_crash_tracker=SlidingWindowCounter(window_seconds=300, threshold=5),
            recovery_timeout_seconds=600,
            max_simultaneous_bad_nodes=2,
            on_main_job_new_run=None,
            rank_pids_provider=None,
            controller_exporter=None,
            on_recovery_duration=None,
            registration_grace_ticks=5,
        )
        ctx = MainContext(
            shared=shared,
            tick_count=10,
            run_start_tick=0,
            job_status=JobStatus.RUNNING,
            node_metadata=metadata,
        )
        actuator = AsyncMock(spec=SubsystemActuatorProtocol)
        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(actuator=actuator),
        )

        result = _build_recovery_context(
            spec=spec,
            context=ctx,
            trigger=TriggerType.CRASH,
            recovery_start_time=datetime.now(timezone.utc),
            restart_stepper=AsyncMock(),
        )

        assert result.restart_context.node_metadata == metadata
        assert result.restart_context.node_metadata["ray-uuid-abc"]["k8s_node_name"] == "gke-node-01"


class TestSharedDepsGrouping:
    """MainContext used to carry ~18 flat fields mixing stable deps with
    per-tick data. Now stable deps are grouped in SharedDeps so MainContext
    is cleaner and per-tick data is clearly separated.
    """

    def test_main_context_has_shared_deps_and_per_tick_fields(self) -> None:
        ctx = _make_main_context()
        assert hasattr(ctx, "shared")
        assert hasattr(ctx, "tick_count")
        assert hasattr(ctx, "job_status")
        assert hasattr(ctx, "node_metadata")

    def test_shared_deps_contains_stable_fields(self) -> None:
        ctx = _make_main_context()
        shared = ctx.shared
        assert shared.main_job is not None
        assert shared.metric_store is not None
        assert shared.recovery_timeout_seconds == 600
        assert shared.max_simultaneous_bad_nodes == 2

    def test_context_factories_read_from_shared(self) -> None:
        """build_subsystem_context reads notifier, metric_store, etc. from ctx.shared."""
        ctx = _make_main_context()
        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(actuator=AsyncMock(spec=SubsystemActuatorProtocol)),
        )
        sub_ctx = build_subsystem_context(
            spec=spec,
            context=ctx,
            recovery_stepper=AsyncMock(),
            restart_stepper=AsyncMock(),
        )
        assert sub_ctx.notifier is ctx.shared.notifier
        assert sub_ctx.metric_store is ctx.shared.metric_store
