import json
from pathlib import Path

import pytest
from tests.fast.utils.observability_fixtures import _LoggingConfig, isolated_observability_logging

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger

__all__ = ["isolated_observability_logging"]


@pytest.mark.usefixtures("isolated_observability_logging")
class TestTypedEventLogConfiguration:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_typed_event_directory_controls_real_event_log_creation(self, tmp_path: Path, enabled: bool) -> None:
        """Typed event-directory settings either leave logging disabled or persist events under the process name."""
        directory = tmp_path / "events"
        config = _LoggingConfig(
            save_debug_event_data=str(directory) if enabled else None, ci_enable_metrics_capture=False
        )
        source = SimpleProcessIdentity(component="main")

        configure_logger(config, source=source, report_env=False)

        assert is_event_logger_initialized() is enabled
        if enabled:
            get_event_logger().log(MetricEvent, {"metrics": {"score": 3}}, print_log=False)
            [path] = directory.iterdir()
            payload = json.loads(path.read_text())
            assert path.name == f"{source.to_name()}.jsonl"
            assert payload["metrics"] == {"score": 3}
            assert payload["source"] == source.model_dump(mode="json")
        else:
            assert not directory.exists()

    def test_later_configuration_preserves_the_installed_event_logger(self, tmp_path: Path) -> None:
        """Reconfiguring a process must not redirect its already installed event stream."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        source = SimpleProcessIdentity(component="main")
        configure_logger(
            _LoggingConfig(save_debug_event_data=str(first), ci_enable_metrics_capture=False),
            source=source,
            report_env=False,
        )
        installed = get_event_logger()

        configure_logger(
            _LoggingConfig(save_debug_event_data=str(second), ci_enable_metrics_capture=False),
            source=source,
            report_env=False,
        )
        get_event_logger().log(MetricEvent, {"metrics": {"score": 7}}, print_log=False)

        assert get_event_logger() is installed
        assert json.loads((first / f"{source.to_name()}.jsonl").read_text())["metrics"] == {"score": 7}
        assert not second.exists()
