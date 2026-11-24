import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path

from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration

from miles.utils import logging_utils
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestLoggerSnapshotCapture:
    @pytest.mark.parametrize("mode", ["enabled", "not_ci", "disabled", "no_directory"])
    def test_logger_initialization_captures_the_current_process_or_clears_disabled_capture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
    ) -> None:
        """Process logging must bind capture to the current identity and respect each capture opt-out."""
        first = _capture_configuration(monkeypatch=monkeypatch, directory=tmp_path)
        monkeypatch.setattr(logging_utils.logging, "basicConfig", lambda **kwargs: None)
        monkeypatch.setattr(logging_utils, "_STRICT_ASYNC_WARNINGS_INSTALLED", True)
        logging_utils.configure_logger(first, source=SimpleProcessIdentity(component="main"), report_env=False)
        second = Namespace(**(vars(first) | {"config_snapshot_name": "second", "run_uuid": "second-run"}))
        if mode == "not_ci":
            second.ci_test = False
        elif mode == "disabled":
            second.ci_disable_config_snapshot = True
        elif mode == "no_directory":
            monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR)

        logging_utils.configure_logger(
            second, source=SimpleProcessIdentity(component="rollout_executor"), report_env=False
        )

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert sorted(record.context.name for record in records) == (
            ["first", "second"] if mode == "enabled" else ["first"]
        )
        assert all(record.point.stage == "process_config" and record.point.index == 0 for record in records)
        first_record = next(record for record in records if record.context.name == "first")
        assert first_record.context.source == SimpleProcessIdentity(component="main")
        assert first_record.config["args"]["run_uuid"] == "snapshot-run"
        if mode == "enabled":
            second_record = next(record for record in records if record.context.name == "second")
            assert second_record.context.source == SimpleProcessIdentity(component="rollout_executor")
            assert second_record.context.capture_id != first_record.context.capture_id
            assert second_record.config["args"]["run_uuid"] == "second-run"
