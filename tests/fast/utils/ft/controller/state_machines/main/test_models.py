"""Tests for miles.utils.ft.controller.state_machines.main.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.main.models import (
    DetectingAnomaly,
    MainState,
    Recovering,
)
from miles.utils.ft.controller.state_machines.recovery.models import RealtimeChecks
from miles.utils.ft.controller.types import TriggerType


class TestMainStateConstruction:
    def test_detecting_anomaly(self) -> None:
        state = DetectingAnomaly()
        assert isinstance(state, MainState)

    def test_recovering(self) -> None:
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        state = Recovering(
            recovery=RealtimeChecks(),
            trigger=TriggerType.HANG,
            recovery_start_time=now,
        )

        assert state.trigger == TriggerType.HANG
        assert isinstance(state.recovery, RealtimeChecks)


class TestMainStateFrozen:
    def test_detecting_anomaly_frozen(self) -> None:
        from datetime import datetime, timezone

        state = DetectingAnomaly()

        with pytest.raises(ValidationError):
            state.model_config = {}  # type: ignore[misc]
