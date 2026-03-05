from __future__ import annotations

import pytest

import miles.utils.ft.metric_names as mn
from miles.utils.ft.models import ActionType, Decision, RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    FixedDecisionDetector,
    get_sample_value,
    make_test_controller,
    make_test_exporter,
)


class TestEnterRecovery:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_recovery_mode_skips_detectors(self) -> None:
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
