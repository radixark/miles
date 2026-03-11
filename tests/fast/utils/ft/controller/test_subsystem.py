from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.subsystem import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig, SubsystemEntry
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


class TestMonitoringIterationProgressConfig:
    def test_defaults(self) -> None:
        config = MonitoringIterationProgressConfig()

        assert config.mode == "iteration_progress"
        assert config.success_iterations == 10
        assert config.timeout_seconds == 600

    def test_custom_values(self) -> None:
        config = MonitoringIterationProgressConfig(
            success_iterations=20,
            timeout_seconds=1200,
        )

        assert config.success_iterations == 20
        assert config.timeout_seconds == 1200

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringIterationProgressConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_alive_duration_rejected(self) -> None:
        """alive_duration_seconds belongs to MonitoringSustainedAliveConfig, not here."""
        with pytest.raises(ValidationError):
            MonitoringIterationProgressConfig(alive_duration_seconds=180)  # type: ignore[call-arg]


class TestMonitoringSustainedAliveConfig:
    def test_defaults(self) -> None:
        config = MonitoringSustainedAliveConfig()

        assert config.mode == "sustained_alive"
        assert config.alive_duration_seconds == 180
        assert config.timeout_seconds == 600

    def test_custom_values(self) -> None:
        config = MonitoringSustainedAliveConfig(
            alive_duration_seconds=300,
            timeout_seconds=900,
        )

        assert config.alive_duration_seconds == 300
        assert config.timeout_seconds == 900

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringSustainedAliveConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_success_iterations_rejected(self) -> None:
        """success_iterations belongs to MonitoringIterationProgressConfig, not here."""
        with pytest.raises(ValidationError):
            MonitoringSustainedAliveConfig(success_iterations=10)  # type: ignore[call-arg]


class TestSubsystemEntry:
    def test_defaults(self) -> None:
        """Default detectors, monitoring_config, and get_active_node_ids are sensible."""
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=MonitoringIterationProgressConfig(), stepper=stepper)

        entry = SubsystemEntry(
            name="training",
            state_machine=sm,
            actuator=AsyncMock(),
        )

        assert entry.detectors == []
        assert isinstance(entry.monitoring_config, MonitoringIterationProgressConfig)
        assert entry.get_active_node_ids() == set()

    def test_custom_get_active_node_ids(self) -> None:
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=MonitoringIterationProgressConfig(), stepper=stepper)
        nodes = {"node-1", "node-2"}

        entry = SubsystemEntry(
            name="rollout",
            state_machine=sm,
            actuator=AsyncMock(),
            get_active_node_ids=lambda: nodes,
        )

        assert entry.get_active_node_ids() == {"node-1", "node-2"}
