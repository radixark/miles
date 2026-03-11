import pytest
from pydantic import ValidationError

from miles.utils.ft.controller.subsystem import MonitoringConfig


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
