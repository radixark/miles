"""Integration tests: Controller → Recovery → DiagnosticScheduler → NodeAgent.

Tests the full diagnostic pipeline through the Controller's recovery flow.
Note: CHECK_ALERTS phase is bypassed (set directly to DIAGNOSING) due to
pre-existing instant_query gap in MiniPrometheus.
"""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.models import RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    ControllerTestHarness,
    make_fake_agents,
    make_test_controller,
)


def _enter_recovery_and_skip_to_diagnosing(
    harness: ControllerTestHarness,
    scheduler: DiagnosticScheduler,
) -> None:
    """Helper: create a RecoveryOrchestrator already in DIAGNOSING phase."""
    from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator

    orch = RecoveryOrchestrator(
        trigger="crash",
        node_manager=harness.node_manager,
        training_job=harness.training_job,
        metric_store=harness.metric_store,
        mini_wandb=harness.mini_wandb,
        notifier=harness.notifier,
        diagnostic_scheduler=scheduler,
        controller_exporter=harness.controller_exporter,
    )
    orch._context.phase = RecoveryPhase.DIAGNOSING
    harness.controller._recovery_orchestrator = orch


class TestDiagnosticPipelineWithBadNode:
    """Diagnostics find bad node → EVICT_AND_RESTART."""

    @pytest.mark.asyncio
    async def test_diagnose_evict_bad_node(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents,
            pipeline=["gpu"],
        )

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        for node_id, agent in agents.items():
            harness.controller.register_agent(node_id, agent)

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None
        assert orch.phase == RecoveryPhase.DIAGNOSING

        # DIAGNOSING → should find node-1 bad → EVICT_AND_RESTART
        await harness.controller._tick()
        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        # Advance to completion
        for _ in range(10):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None
        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")


class TestDiagnosticPipelineAllPass:
    """All diagnostics pass → NOTIFY_HUMAN."""

    @pytest.mark.asyncio
    async def test_all_pass_leads_to_notify(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True},
            "node-1": {"gpu": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents,
            pipeline=["gpu"],
        )

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        for node_id, agent in agents.items():
            harness.controller.register_agent(node_id, agent)

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        # DIAGNOSING: all pass → NOTIFY
        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY

        # NOTIFY → DONE
        await harness.controller._tick()

        for _ in range(5):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-1")


class TestDiagnosticPipelineEmptyPipeline:
    """Empty pipeline (no diagnostics) → NOTIFY (backward compat with stub)."""

    @pytest.mark.asyncio
    async def test_empty_pipeline_notifies(self) -> None:
        scheduler = DiagnosticScheduler(agents={}, pipeline=[])

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None
        assert orch.phase == RecoveryPhase.DIAGNOSING

        # Empty pipeline → NOTIFY
        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY


class TestDiagnosticPipelineInterMachine:
    """Inter-machine step catches bad node through cross-comparison."""

    @pytest.mark.asyncio
    async def test_inter_machine_catches_bad_node(self) -> None:
        # 3 nodes, gpu+intra pass for all, inter-machine isolates node-1
        # node-1 has inter_machine=False → pairs (node-0,node-1) and
        # (node-1,node-2) fail, (node-2,node-0) passes
        # failure_count: node-0=1, node-1=2, node-2=1 → node-1 is bad
        agents = make_fake_agents({
            "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
            "node-1": {"gpu": True, "intra_machine": True, "inter_machine": False},
            "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents,
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        for node_id, agent in agents.items():
            harness.controller.register_agent(node_id, agent)

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        await harness.controller._tick()
        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        for _ in range(10):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None
        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-2")

    @pytest.mark.asyncio
    async def test_full_pipeline_all_pass(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
            "node-1": {"gpu": True, "intra_machine": True, "inter_machine": True},
            "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
        })
        scheduler = DiagnosticScheduler(
            agents=agents,
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        for node_id, agent in agents.items():
            harness.controller.register_agent(node_id, agent)

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY

        await harness.controller._tick()

        for _ in range(5):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-2")


class TestDiagnosticPipelineMultiStep:
    """Multi-step pipeline catches bad node at second step."""

    @pytest.mark.asyncio
    async def test_multi_step_second_step_catches(self) -> None:
        agents = make_fake_agents({
            "node-0": {"gpu": True, "intra": True},
            "node-1": {"gpu": True, "intra": False},
        })
        scheduler = DiagnosticScheduler(
            agents=agents,
            pipeline=["gpu", "intra"],
        )

        harness = make_test_controller(
            status_sequence=[JobStatus.RUNNING] * 50,
            diagnostic_scheduler=scheduler,
        )

        for node_id, agent in agents.items():
            harness.controller.register_agent(node_id, agent)

        _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
        orch = harness.controller._recovery_orchestrator
        assert orch is not None

        # DIAGNOSING → gpu passes, intra fails node-1 → EVICT_AND_RESTART
        await harness.controller._tick()
        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        for _ in range(10):
            if harness.controller._recovery_orchestrator is None:
                break
            await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is None
        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")
