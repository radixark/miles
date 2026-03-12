"""Tests for miles.utils.ft.controller.state_machines.restart.models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.restart.models import (
    EvictingSt,
    MonitoringIterationProgressConfig,
    MonitoringProgressSt,
    MonitoringSustainedAliveConfig,
    RestartDoneSt,
    RestartFailedSt,
    RestartState,
    StoppingAndRestartingSt,
)


class TestRestartStateConstruction:
    def test_evicting(self) -> None:
        state = EvictingSt(bad_node_ids=["node-0"])
        assert state.bad_node_ids == ("node-0",)

    def test_evicting_default_bad_nodes(self) -> None:
        state = EvictingSt()
        assert state.bad_node_ids == ()

    def test_stopping_and_restarting_defaults(self) -> None:
        state = StoppingAndRestartingSt()
        assert state.submitted is False
        assert state.submit_time is None

    def test_stopping_and_restarting_with_submit_time(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = StoppingAndRestartingSt(submitted=True, submit_time=now)
        assert state.submitted is True
        assert state.submit_time == now

    def test_monitoring_progress(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = MonitoringProgressSt(start_time=now, base_iteration=100)
        assert state.start_time == now
        assert state.base_iteration == 100

    def test_restart_done(self) -> None:
        state = RestartDoneSt()
        assert isinstance(state, RestartState)

    def test_restart_failed(self) -> None:
        state = RestartFailedSt()
        assert isinstance(state, RestartState)


class TestStoppingAndRestartingValidator:
    def test_submitted_true_without_submit_time_raises(self) -> None:
        """Previously submitted=True with submit_time=None was silently accepted,
        causing the timeout check in _poll() to be skipped and the handler
        to hang indefinitely."""
        with pytest.raises(ValidationError, match="submit_time must be set"):
            StoppingAndRestartingSt(submitted=True, submit_time=None)

    def test_submitted_false_without_submit_time_ok(self) -> None:
        state = StoppingAndRestartingSt(submitted=False, submit_time=None)
        assert state.submitted is False

    def test_submitted_true_with_submit_time_ok(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = StoppingAndRestartingSt(submitted=True, submit_time=now)
        assert state.submitted is True
        assert state.submit_time == now


class TestMonitoringConfigValidation:
    """Config fields lacked ge=0 validation, so negative values could cause
    instant success (iterations=0) or instant timeout (timeout_seconds=0)."""

    def test_iteration_progress_negative_success_iterations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringIterationProgressConfig(success_iterations=-1)

    def test_iteration_progress_zero_success_iterations_allowed(self) -> None:
        cfg = MonitoringIterationProgressConfig(success_iterations=0)
        assert cfg.success_iterations == 0

    def test_iteration_progress_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringIterationProgressConfig(timeout_seconds=-1)

    def test_sustained_alive_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringSustainedAliveConfig(alive_duration_seconds=-1)

    def test_sustained_alive_zero_duration_allowed(self) -> None:
        cfg = MonitoringSustainedAliveConfig(alive_duration_seconds=0)
        assert cfg.alive_duration_seconds == 0

    def test_sustained_alive_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringSustainedAliveConfig(timeout_seconds=-1)


class TestRestartStateFrozen:
    def test_evicting_frozen(self) -> None:
        state = EvictingSt(bad_node_ids=["node-0"])
        with pytest.raises(ValidationError):
            state.bad_node_ids = ["node-1"]  # type: ignore[misc]

    def test_stopping_and_restarting_frozen(self) -> None:
        state = StoppingAndRestartingSt()
        with pytest.raises(ValidationError):
            state.submitted = True  # type: ignore[misc]
