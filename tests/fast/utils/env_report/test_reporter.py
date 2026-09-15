import logging
from pathlib import Path
from unittest.mock import patch

from tests.fast.utils.env_report.conftest import make_args

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import EnvReportEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.env_report.collector import collect_env_report, collect_env_report_snapshot
from miles.utils.env_report.reporter import _log_env_report, start_env_reporting


class TestLogEnvReport:
    def _log(self, **overrides) -> None:
        _log_env_report(report=collect_env_report(snapshot=collect_env_report_snapshot(make_args(**overrides))))

    def test_writes_one_event_the_analyzer_can_read_back(self, mocked_pip_inspect, event_log_dir: Path) -> None:
        """The report is stored as a normal event, so replaying a run's jsonl recovers its environment."""
        self._log(lr=1.0)

        events = read_events(event_log_dir)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, EnvReportEvent)
        assert event.source == SimpleProcessIdentity(component="main")
        assert event.report.process.args.values["lr"] == 1.0
        assert event.report.process.hostname

    def test_summarises_the_report_on_stdout_instead_of_dumping_it(
        self, mocked_pip_inspect, event_log_dir: Path, caplog
    ) -> None:
        """A full report is tens of kilobytes; logging it per process would drown the logs."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log(lr=1.0)

        assert "op=env_report" in caplog.text
        assert "num_packages=4" in caplog.text
        assert "PYTHONUNBUFFERED" not in caplog.text

    def test_summarises_the_report_when_no_event_logger_is_configured(
        self, mocked_pip_inspect, without_event_logger, caplog
    ) -> None:
        """A run without an event dir still leaves a trace instead of silently dropping the report."""
        with caplog.at_level(logging.INFO, logger="miles.utils.env_report.reporter"):
            self._log()

        assert "op=env_report" in caplog.text
        assert "stored=false" in caplog.text


class TestEnvReporter:
    def test_reports_the_environment_of_the_process_that_starts_it(
        self, mocked_pip_inspect, event_log_dir: Path
    ) -> None:
        """Starting the reporting is the whole contract; nothing else in the process asks for a report."""
        start_env_reporting(make_args(lr=1.0))

        events = read_events(event_log_dir)
        assert len(events) == 1
        assert events[0].report.process.args.values["lr"] == 1.0

    def test_a_failing_collection_only_warns(self, event_log_dir: Path, caplog) -> None:
        """An audit that cannot read its own environment must not end the process's startup."""
        with patch("miles.utils.env_report.reporter.collect_env_report_snapshot", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="miles.utils.env_report.reporter"):
                start_env_reporting(make_args())

        assert "Failed to log the env report" in caplog.text
        assert read_events(event_log_dir) == []
