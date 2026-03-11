from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import AlwaysEnterRecoveryDetector, make_test_controller
from tests.fast.utils.ft.utils.controller_fakes import get_training_subsystem_state

from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt, RecoveringSt
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType


def _force_recovery_complete(harness) -> None:
    """Force the machine back to DetectingAnomaly and re-register a rank
    so that detectors will fire on the next tick."""
    from tests.fast.utils.ft.utils.controller_fakes import set_training_subsystem_state
    set_training_subsystem_state(harness.controller, DetectingAnomalySt())
    harness.controller._activate_run("recovery-done")
    harness.controller.training_rank_roster.rank_placement[0] = "node-0"


class TestRecoveryCooldown:
    @pytest.mark.anyio
    async def test_third_crash_recovery_escalates_to_notify_human(self) -> None:
        detector = AlwaysEnterRecoveryDetector(reason="training crashed")
        harness = make_test_controller(
            detectors=[detector],
            recovery_cooldown_max_count=3,
        )

        # Step 1: first recovery
        await harness.controller._tick()
        assert isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)
        _force_recovery_complete(harness)

        # Step 2: second recovery
        await harness.controller._tick()
        assert isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)
        _force_recovery_complete(harness)

        # Step 3: third attempt throttled
        await harness.controller._tick()
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, content, severity = harness.notifier.calls[0]
        assert "Recovery cooldown" in content
        assert "CRASH" in content

    @pytest.mark.anyio
    async def test_all_triggers_counted_globally(self) -> None:
        """After 2 crashes, a HANG trigger also counts toward the global cooldown."""
        crash_detector = AlwaysEnterRecoveryDetector(reason="crash")
        harness = make_test_controller(
            detectors=[crash_detector],
            recovery_cooldown_max_count=3,
        )

        await harness.controller._tick()
        _force_recovery_complete(harness)
        await harness.controller._tick()
        _force_recovery_complete(harness)

        crash_detector._decision = Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=TriggerType.HANG,
            reason="hang",
        )
        await harness.controller._tick()
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)

    @pytest.mark.anyio
    async def test_recovery_within_cooldown_window_counted(self) -> None:
        detector = AlwaysEnterRecoveryDetector(reason="crash")
        harness = make_test_controller(
            detectors=[detector],
            recovery_cooldown_max_count=2,
        )

        await harness.controller._tick()
        _force_recovery_complete(harness)

        await harness.controller._tick()
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
