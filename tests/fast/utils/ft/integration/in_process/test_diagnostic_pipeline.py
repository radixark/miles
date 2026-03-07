"""Integration tests: Controller → Recovery → DiagnosticOrchestrator → NodeAgent.

Tests the full diagnostic pipeline through the Controller's recovery flow
by injecting StopTimeDiagnostics state directly into the state machine.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator
from miles.utils.ft.controller.main_state_machine import Recovering
from miles.utils.ft.controller.recovery.recovery_stepper import StopTimeDiagnostics
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    ControllerTestHarness,
    make_fake_agents,
    make_test_controller,
)


def _make_diagnostic_test_env(
    node_results: dict[str, dict[str, bool]],
    pipeline: list[str],
) -> ControllerTestHarness:
    agents = make_fake_agents(node_results)
    orchestrator = DiagnosticOrchestrator(agents=agents, pipeline=pipeline)
    harness = make_test_controller(
        status_sequence=[JobStatus.RUNNING] * 50,
        diagnostic_orchestrator=orchestrator,
    )
    for node_id, agent in agents.items():
        harness.controller.register_node_agent(node_id, agent)

    harness.controller._state_machine._state = Recovering(
        recovery=StopTimeDiagnostics(),
        trigger="crash",
        recovery_start_time=datetime.now(timezone.utc),
    )
    return harness


class TestDiagnosticPipelineWithBadNode:
    """Diagnostics find bad node → EvictingAndRestarting."""

    @pytest.mark.anyio
    async def test_diagnose_evict_bad_node(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={"node-0": {"gpu": True}, "node-1": {"gpu": False}},
            pipeline=["gpu"],
        )

        # StopTimeDiagnostics → find node-1 bad → EvictingAndRestarting → evict
        await harness.controller._tick()

        assert harness.node_manager.was_ever_marked_bad("node-1")
        assert not harness.node_manager.was_ever_marked_bad("node-0")
        assert isinstance(harness.controller._state_machine.state, Recovering)


class TestDiagnosticPipelineAllPass:
    """All diagnostics pass → NotifyHumans → RecoveryDone."""

    @pytest.mark.anyio
    async def test_all_pass_leads_to_notify(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={"node-0": {"gpu": True}, "node-1": {"gpu": True}},
            pipeline=["gpu"],
        )

        # StopTimeDiagnostics → all pass → NotifyHumans → RecoveryDone → DetectingAnomaly
        await harness.controller._tick()

        assert not isinstance(harness.controller._state_machine.state, Recovering)
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.was_ever_marked_bad("node-0")
        assert not harness.node_manager.was_ever_marked_bad("node-1")


class TestDiagnosticPipelineEmptyPipeline:
    """Empty pipeline (no diagnostics) → NotifyHumans."""

    @pytest.mark.anyio
    async def test_empty_pipeline_notifies(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={},
            pipeline=[],
        )

        # Empty pipeline → NotifyHumans → RecoveryDone → DetectingAnomaly
        await harness.controller._tick()

        assert not isinstance(harness.controller._state_machine.state, Recovering)
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1


class TestDiagnosticPipelineInterMachine:
    """Inter-machine step catches bad node through cross-comparison."""

    @pytest.mark.anyio
    async def test_inter_machine_catches_bad_node(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-1": {"gpu": True, "intra_machine": True, "inter_machine": False},
                "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
            },
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        await harness.controller._tick()

        assert harness.node_manager.was_ever_marked_bad("node-1")
        assert not harness.node_manager.was_ever_marked_bad("node-0")
        assert not harness.node_manager.was_ever_marked_bad("node-2")

    @pytest.mark.anyio
    async def test_full_pipeline_all_pass(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-1": {"gpu": True, "intra_machine": True, "inter_machine": True},
                "node-2": {"gpu": True, "intra_machine": True, "inter_machine": True},
            },
            pipeline=["gpu", "intra_machine", "inter_machine"],
        )

        # All pass → NotifyHumans → RecoveryDone → DetectingAnomaly
        await harness.controller._tick()

        assert not isinstance(harness.controller._state_machine.state, Recovering)
        assert harness.notifier is not None
        assert len(harness.notifier.calls) >= 1
        assert not harness.node_manager.was_ever_marked_bad("node-0")
        assert not harness.node_manager.was_ever_marked_bad("node-1")
        assert not harness.node_manager.was_ever_marked_bad("node-2")


class TestDiagnosticPipelineMultiStep:
    """Multi-step pipeline catches bad node at second step."""

    @pytest.mark.anyio
    async def test_multi_step_second_step_catches(self) -> None:
        harness = _make_diagnostic_test_env(
            node_results={
                "node-0": {"gpu": True, "intra": True},
                "node-1": {"gpu": True, "intra": False},
            },
            pipeline=["gpu", "intra"],
        )

        # gpu passes all, intra fails node-1 → EvictingAndRestarting → evict
        await harness.controller._tick()

        assert harness.node_manager.was_ever_marked_bad("node-1")
        assert not harness.node_manager.was_ever_marked_bad("node-0")
