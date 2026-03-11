from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.subsystem import IterationProgressConfig, SustainedAliveConfig, SubsystemEntry
from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


class TestIterationProgressConfig:
    def test_defaults(self) -> None:
        config = IterationProgressConfig()

        assert config.mode == "iteration_progress"
        assert config.success_iterations == 10
        assert config.timeout_seconds == 600

    def test_custom_values(self) -> None:
        config = IterationProgressConfig(
            success_iterations=20,
            timeout_seconds=1200,
        )

        assert config.success_iterations == 20
        assert config.timeout_seconds == 1200

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IterationProgressConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_alive_duration_rejected(self) -> None:
        """alive_duration_seconds belongs to SustainedAliveConfig, not here."""
        with pytest.raises(ValidationError):
            IterationProgressConfig(alive_duration_seconds=180)  # type: ignore[call-arg]


class TestSustainedAliveConfig:
    def test_defaults(self) -> None:
        config = SustainedAliveConfig()

        assert config.mode == "sustained_alive"
        assert config.alive_duration_seconds == 180
        assert config.timeout_seconds == 600

    def test_custom_values(self) -> None:
        config = SustainedAliveConfig(
            alive_duration_seconds=300,
            timeout_seconds=900,
        )

        assert config.alive_duration_seconds == 300
        assert config.timeout_seconds == 900

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SustainedAliveConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_success_iterations_rejected(self) -> None:
        """success_iterations belongs to IterationProgressConfig, not here."""
        with pytest.raises(ValidationError):
            SustainedAliveConfig(success_iterations=10)  # type: ignore[call-arg]


class TestSubsystemEntry:
    def test_defaults(self) -> None:
        """Default detectors, monitoring_config, and get_active_node_ids are sensible."""
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=IterationProgressConfig(), stepper=stepper)

        entry = SubsystemEntry(
            name="training",
            state_machine=sm,
            actuator=AsyncMock(),
        )

        assert entry.detectors == []
        assert isinstance(entry.monitoring_config, IterationProgressConfig)
        assert entry.get_active_node_ids() == set()

    def test_custom_get_active_node_ids(self) -> None:
        stepper = StateMachineStepper(handler_map={})
        sm = StateMachine(initial_state=IterationProgressConfig(), stepper=stepper)
        nodes = {"node-1", "node-2"}

        entry = SubsystemEntry(
            name="rollout",
            state_machine=sm,
            actuator=AsyncMock(),
            get_active_node_ids=lambda: nodes,
        )

        assert entry.get_active_node_ids() == {"node-1", "node-2"}
