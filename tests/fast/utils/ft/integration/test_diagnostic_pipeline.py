"""Integration tests: Controller → Recovery → DiagnosticScheduler → NodeAgent.

Tests the full diagnostic pipeline through the Controller's recovery flow.
Note: CHECK_ALERTS phase is bypassed (set directly to DIAGNOSING) due to
pre-existing instant_query gap in MiniPrometheus.
"""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.diagnostics.scheduler import DiagnosticScheduler
from miles.utils.ft.controller.recovery_orchestrator import RecoveryOrchestrator
from miles.utils.ft.models import RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    ControllerTestHarness,
    advance_until_recovery_complete,
    make_fake_agents,
    make_test_controller,
)


def _enter_recovery_and_skip_to_diagnosing(
    harness: ControllerTestHarness,
    scheduler: DiagnosticScheduler,
) -> None:
    """Helper: create a RecoveryOrchestrator already in DIAGNOSING phase."""
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
    harness.controller._recovery_manager._orchestrator = orch


def _make_diagnostic_test_env(
    node_results: dict[str, dict[str, bool]],
    pipeline: list[str],
) -> tuple[ControllerTestHarness, RecoveryOrchestrator]:
    agents = make_fake_agents(node_results)
    scheduler = DiagnosticScheduler(agents=agents, pipeline=pipeline)
    harness = make_test_controller(
        status_sequence=[JobStatus.RUNNING] * 50,
        diagnostic_scheduler=scheduler,
    )
    for node_id, agent in agents.items():
        harness.rank_registry.register_agent(node_id, agent)
    _enter_recovery_and_skip_to_diagnosing(harness, scheduler)
    orch = harness.controller.recovery_manager.orchestrator
    assert orch is not None
    return harness, orch


class TestDiagnosticPipelineWithBadNode:
    """Diagnostics find bad node → EVICT_AND_RESTART."""

    @pytest.mark.anyio
    async def test_diagnose_evict_bad_node(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={"node-0": {"gpu": True}, "node-1": {"gpu": False}},
            pipeline=["gpu"],
        )
        assert orch.phase == RecoveryPhase.DIAGNOSING

        # DIAGNOSING → should find node-1 bad → EVICT_AND_RESTART
        await harness.controller._tick()
        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        # Advance to completion
        await advance_until_recovery_complete(harness)

        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")


class TestDiagnosticPipelineAllPass:
    """All diagnostics pass → NOTIFY_HUMAN."""

    @pytest.mark.anyio
    async def test_all_pass_leads_to_notify(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={"node-0": {"gpu": True}, "node-1": {"gpu": True}},
            pipeline=["gpu"],
        )

        # DIAGNOSING: all pass → NOTIFY
        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY

        # NOTIFY → DONE
        await harness.controller._tick()

        await advance_until_recovery_complete(harness)

        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-1")


class TestDiagnosticPipelineEmptyPipeline:
    """Empty pipeline (no diagnostics) → NOTIFY (backward compat with stub)."""

    @pytest.mark.anyio
    async def test_empty_pipeline_notifies(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={},
            pipeline=[],
        )
        assert orch.phase == RecoveryPhase.DIAGNOSING

        # Empty pipeline → NOTIFY
        await harness.controller._tick()
        assert orch.phase == RecoveryPhase.NOTIFY


class TestDiagnosticPipelineInterMachine:
    """Inter-machine step catches bad node through cross-comparison."""

    @pytest.mark.anyio
    async def test_inter_machine_catches_bad_node(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-1": {"gpu": True, "intra_machine": True, "inter_machine": False},
                "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
            },
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        await harness.controller._tick()

        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        await advance_until_recovery_complete(harness)

        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-2")

    @pytest.mark.anyio
    async def test_full_pipeline_all_pass(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-1": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
            },
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        await harness.controller._tick()

        assert orch.phase == RecoveryPhase.NOTIFY

        await harness.controller._tick()

        await advance_until_recovery_complete(harness)

        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.is_node_bad("node-0")
        assert not harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-2")


class TestDiagnosticPipelineMultiStep:
    """Multi-step pipeline catches bad node at second step."""

    @pytest.mark.anyio
    async def test_multi_step_second_step_catches(self) -> None:
        harness, orch = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": False},
            },
            pipeline=["gpu", "intra"],
        )

        # DIAGNOSING → gpu passes, intra fails node-1 → EVICT_AND_RESTART
        await harness.controller._tick()
        assert orch.phase in (
            RecoveryPhase.EVICT_AND_RESTART, RecoveryPhase.DONE,
        )

        await advance_until_recovery_complete(harness)

        assert harness.node_manager.is_node_bad("node-1")
        assert not harness.node_manager.is_node_bad("node-0")
