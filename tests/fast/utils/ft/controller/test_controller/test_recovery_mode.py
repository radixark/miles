from __future__ import annotations

import pytest

import miles.utils.ft.metric_names as mn
from miles.utils.ft.models import ActionType, Decision, RecoveryPhase
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    CriticalFixedDecisionDetector,
    FixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)


class TestEnterRecovery:
    @pytest.mark.anyio
    async def test_creates_recovery_orchestrator(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        assert harness.controller._recovery_orchestrator is None

        await harness.controller._tick()

        assert harness.controller._recovery_orchestrator is not None

    @pytest.mark.anyio
    async def test_recovery_mode_skips_non_critical_detectors(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        initial_count = detector.call_count

        await harness.controller._tick()
        assert detector.call_count == initial_count

    @pytest.mark.anyio
    async def test_critical_detector_runs_during_recovery(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        critical = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-new-bad"],
            reason="critical hw fault during recovery",
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical])

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is not None
        assert critical.call_count == 0

        await harness.controller._tick()
        assert critical.call_count == 1
        assert "node-new-bad" in harness.controller._recovery_orchestrator.bad_node_ids

    @pytest.mark.anyio
    async def test_critical_detector_no_duplicate_bad_nodes(self) -> None:
        enter_recovery = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        critical = CriticalFixedDecisionDetector(decision=Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-1"],
            reason="same node again",
        ))
        harness = make_test_controller(detectors=[enter_recovery, critical])

        await harness.controller._tick()
        orch = harness.controller._recovery_orchestrator
        assert orch is not None
        orch._context.bad_node_ids.append("node-1")

        await harness.controller._tick()
        assert orch.bad_node_ids.count("node-1") == 1

    @pytest.mark.anyio
    async def test_recovery_complete_returns_to_monitoring(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(
            detectors=[detector],
            status_sequence=[JobStatus.RUNNING],
        )

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is not None

        harness.controller._recovery_orchestrator._context.phase = RecoveryPhase.DONE

        await harness.controller._tick()
        assert harness.controller._recovery_orchestrator is None

    @pytest.mark.anyio
    async def test_exporter_mode_reflects_recovery(self) -> None:
        registry, exporter = make_test_exporter()
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger="crash",
            reason="test recovery",
        ))
        harness = make_test_controller(
            detectors=[detector],
            controller_exporter=exporter,
        )

        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

        await harness.controller._tick()

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0
