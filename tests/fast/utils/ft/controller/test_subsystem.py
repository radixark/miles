from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.state_machines.restart.models import MonitoringIterationProgressConfig, MonitoringSustainedAliveConfig
from miles.utils.ft.controller.subsystem_hub import SubsystemConfig, SubsystemRuntime, SubsystemSpec


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


class TestSubsystemSpec:
    """SubsystemConfig used to mix static config fields (restart_mode, detectors)
    with runtime state (actuator, cooldown, get_active_node_ids). They are now
    separated into SubsystemConfig (pure config) and SubsystemRuntime (runtime deps),
    wrapped by SubsystemSpec.
    """

    def test_defaults(self) -> None:
        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(actuator=AsyncMock()),
        )

        assert spec.config.detectors == []
        assert isinstance(spec.config.monitoring_config, MonitoringIterationProgressConfig)
        assert spec.runtime.get_active_node_ids() == frozenset()

    def test_custom_get_active_node_ids(self) -> None:
        nodes = frozenset({"node-1", "node-2"})

        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(
                actuator=AsyncMock(),
                get_active_node_ids=lambda: nodes,
            ),
        )

        assert spec.runtime.get_active_node_ids() == frozenset({"node-1", "node-2"})

    def test_config_and_runtime_are_separate(self) -> None:
        """Static config should not contain runtime objects, and vice versa."""
        spec = SubsystemSpec(
            config=SubsystemConfig(),
            runtime=SubsystemRuntime(actuator=AsyncMock()),
        )

        assert not hasattr(spec.config, "actuator")
        assert not hasattr(spec.config, "cooldown")
        assert not hasattr(spec.config, "get_active_node_ids")

        assert not hasattr(spec.runtime, "restart_mode")
        assert not hasattr(spec.runtime, "detectors")
        assert not hasattr(spec.runtime, "monitoring_config")
