import logging
from dataclasses import dataclass
from typing import Literal

import pytest

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import (
    ExecuteTrainConfig,
    resolve_extra_env_vars,
    resolve_hardware,
)


@dataclass
class _HardwareConfig(ExecuteTrainConfig):
    hardware: Literal["auto", "H100"] = "auto"


class TestResolveHardware:
    @pytest.mark.parametrize("configured", [False, True])
    def test_hardware_detection_is_visible_without_changing_logging(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], configured: bool
    ) -> None:
        """Hardware detection writes to stderr without changing application logging."""
        root = logging.getLogger()
        handlers = [logging.NullHandler()] if configured else []
        expected_handlers = tuple(handlers)
        monkeypatch.setattr(root, "handlers", handlers)
        monkeypatch.setattr(root, "level", logging.WARNING)
        monkeypatch.setattr(base_backend, "detect_hardware", lambda: "H100")

        assert resolve_hardware(_HardwareConfig()) == "H100"
        captured = capsys.readouterr()
        assert captured.err == "detected --hardware H100\n"
        assert captured.out == ""
        assert root.handlers is handlers
        assert tuple(root.handlers) == expected_handlers
        assert root.level == logging.WARNING

    def test_supported_explicit_value_bypasses_detection_while_auto_uses_it(self, monkeypatch):
        """Explicit hardware bypasses detection, while auto resolves to a supported detected profile."""
        detected: list[None] = []

        def detect_hardware() -> str:
            detected.append(None)
            return "H100"

        monkeypatch.setattr(base_backend, "detect_hardware", detect_hardware)

        assert resolve_hardware(_HardwareConfig(hardware="H100")) == "H100"
        assert detected == []
        assert resolve_hardware(_HardwareConfig(hardware="auto")) == "H100"
        assert detected == [None]

    @pytest.mark.parametrize(
        ("configured_hardware", "detected_hardware"),
        [("unsupported", "H100"), ("auto", "unsupported")],
    )
    def test_unsupported_explicit_or_detected_value_is_rejected(
        self, configured_hardware: str, detected_hardware: str, monkeypatch
    ):
        """Neither explicit nor detected hardware may escape the config's supported profile literal."""
        monkeypatch.setattr(base_backend, "detect_hardware", lambda: detected_hardware)

        with pytest.raises(AssertionError, match="has no verified profile"):
            resolve_hardware(_HardwareConfig(hardware=configured_hardware))


class TestResolveExtraEnvVars:
    def test_config_extra_env_vars_override_the_callers_values(self):
        """Parsed config variables override duplicates while preserving caller-only variables."""
        config = ExecuteTrainConfig(extra_env_vars="SHARED=from_config CONFIG_ONLY=kept")

        resolved = resolve_extra_env_vars(
            extra_env_vars={"SHARED": "from_caller", "CALLER_ONLY": "kept"},
            config=config,
        )

        assert resolved == {
            "SHARED": "from_config",
            "CALLER_ONLY": "kept",
            "CONFIG_ONLY": "kept",
        }
