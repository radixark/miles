"""Tests for miles.utils.ft.controller.state_machines.restart.models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.restart.models import (
    Evicting,
    MonitoringProgress,
    RestartDone,
    RestartFailed,
    RestartState,
    StoppingAndRestarting,
)


class TestRestartStateConstruction:
    def test_evicting(self) -> None:
        state = Evicting(bad_node_ids=["node-0"])
        assert state.bad_node_ids == ["node-0"]

    def test_evicting_default_bad_nodes(self) -> None:
        state = Evicting()
        assert state.bad_node_ids == []

    def test_stopping_and_restarting_defaults(self) -> None:
        state = StoppingAndRestarting()
        assert state.submitted is False
        assert state.submit_time is None

    def test_stopping_and_restarting_with_submit_time(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = StoppingAndRestarting(submitted=True, submit_time=now)
        assert state.submitted is True
        assert state.submit_time == now

    def test_monitoring_progress(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = MonitoringProgress(start_time=now, base_iteration=100)
        assert state.start_time == now
        assert state.base_iteration == 100

    def test_restart_done(self) -> None:
        state = RestartDone()
        assert isinstance(state, RestartState)

    def test_restart_failed(self) -> None:
        state = RestartFailed()
        assert isinstance(state, RestartState)


class TestRestartStateFrozen:
    def test_evicting_frozen(self) -> None:
        state = Evicting(bad_node_ids=["node-0"])
        with pytest.raises(ValidationError):
            state.bad_node_ids = ["node-1"]  # type: ignore[misc]

    def test_stopping_and_restarting_frozen(self) -> None:
        state = StoppingAndRestarting()
        with pytest.raises(ValidationError):
            state.submitted = True  # type: ignore[misc]
