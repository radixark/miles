from __future__ import annotations

import pytest

from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    make_test_controller,
)


class TestRecoveryCooldown:
    @pytest.mark.anyio
    async def test_third_crash_recovery_escalates_to_notify_human(self) -> None:
        detector = AlwaysEnterRecoveryDetector(reason="training crashed")
        harness = make_test_controller(
            detectors=[detector],
            recovery_cooldown_max_count=3,
        )

        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress
        harness.controller._recovery_manager._orchestrator = None

        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress
        harness.controller._recovery_manager._orchestrator = None

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, content, severity = harness.notifier.calls[0]
        assert "Recovery cooldown" in content
        assert TriggerType.CRASH.value in content

    @pytest.mark.anyio
    async def test_different_triggers_tracked_separately(self) -> None:
        crash_detector = AlwaysEnterRecoveryDetector(reason="crash")
        harness = make_test_controller(
            detectors=[crash_detector],
            recovery_cooldown_max_count=3,
        )

        await harness.controller._tick()
        harness.controller._recovery_manager._orchestrator = None
        await harness.controller._tick()
        harness.controller._recovery_manager._orchestrator = None

        crash_detector._decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=TriggerType.HANG,
            reason="hang",
        )
        await harness.controller._tick()
        assert harness.controller._recovery_manager.in_progress

    @pytest.mark.anyio
    async def test_recovery_within_cooldown_window_counted(self) -> None:
        detector = AlwaysEnterRecoveryDetector(reason="crash")
        harness = make_test_controller(
            detectors=[detector],
            recovery_cooldown_max_count=2,
        )

        await harness.controller._tick()
        harness.controller._recovery_manager._orchestrator = None

        await harness.controller._tick()
        assert not harness.controller._recovery_manager.in_progress
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
