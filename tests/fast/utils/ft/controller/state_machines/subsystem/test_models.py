"""Tests for miles.utils.ft.controller.state_machines.subsystem.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.subsystem.models import DetectingAnomalySt, SubsystemState, RecoveringSt
from miles.utils.ft.controller.state_machines.recovery.models import RealtimeChecksSt
from miles.utils.ft.controller.types import TriggerType


class TestSubsystemStateConstruction:
    def test_detecting_anomaly(self) -> None:
        state = DetectingAnomalySt()
        assert isinstance(state, SubsystemState)

    def test_recovering(self) -> None:
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        state = RecoveringSt(
            recovery=RealtimeChecksSt(),
            trigger=TriggerType.HANG,
            recovery_start_time=now,
        )

        assert state.trigger == TriggerType.HANG
        assert isinstance(state.recovery, RealtimeChecksSt)


class TestSubsystemStateFrozen:
    def test_detecting_anomaly_frozen(self) -> None:

        state = DetectingAnomalySt()

        with pytest.raises(ValidationError):
            state.model_config = {}  # type: ignore[misc]
