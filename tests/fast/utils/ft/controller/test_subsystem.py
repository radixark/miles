from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.subsystem import MonitoringConfig, SubsystemEntry
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


class TestMonitoringConfig:
    def test_iteration_progress_defaults(self) -> None:
        config = MonitoringConfig(mode="iteration_progress")

        assert config.mode == "iteration_progress"
        assert config.success_iterations == 10
        assert config.timeout_seconds == 600
        assert config.alive_duration_seconds == 180

    def test_sustained_alive_defaults(self) -> None:
        config = MonitoringConfig(mode="sustained_alive")

        assert config.mode == "sustained_alive"
        assert config.alive_duration_seconds == 180
        assert config.success_iterations == 10
        assert config.timeout_seconds == 600

    def test_custom_values(self) -> None:
        config = MonitoringConfig(
            mode="iteration_progress",
            success_iterations=20,
            timeout_seconds=1200,
        )

        assert config.success_iterations == 20
        assert config.timeout_seconds == 1200

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringConfig(mode="iteration_progress", unknown_field=42)  # type: ignore[call-arg]

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MonitoringConfig(mode="invalid_mode")  # type: ignore[arg-type]


class TestSubsystemEntry:
    def test_defaults(self) -> None:
        """Default detectors, monitoring_config, and get_active_node_ids are sensible."""
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=MonitoringConfig(mode="iteration_progress"), stepper=stepper)

        entry = SubsystemEntry(
            name="training",
            state_machine=sm,
            actuator=AsyncMock(),
        )

        assert entry.detectors == []
        assert entry.monitoring_config.mode == "iteration_progress"
        assert entry.get_active_node_ids() == set()

    def test_custom_get_active_node_ids(self) -> None:
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=MonitoringConfig(mode="iteration_progress"), stepper=stepper)
        nodes = {"node-1", "node-2"}

        entry = SubsystemEntry(
            name="rollout",
            state_machine=sm,
            actuator=AsyncMock(),
            get_active_node_ids=lambda: nodes,
        )

        assert entry.get_active_node_ids() == {"node-1", "node-2"}
