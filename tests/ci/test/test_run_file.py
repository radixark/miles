import json
import logging
import os
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import CI_GATE_RECORD_DIR_ENV
from tests.ci.run_file import app
from tests.ci.test.conftest import _SnapshotFileCase
from typer.testing import CliRunner

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize("existing_directory", [False, True])
def test_child_captures_metrics_without_a_database_store(
    snapshot_file_case: _SnapshotFileCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_directory: bool,
) -> None:
    """Single-file runs retain suite-equivalent metric capture without database access."""
    snapshot_file_case.write_golden(value="actual")
    supplied_directory = tmp_path / "metrics"
    if existing_directory:
        monkeypatch.setenv(CI_GATE_RECORD_DIR_ENV, str(supplied_directory))
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://invalid.invalid/forbidden")

    result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"])

    assert result.exit_code == 0, result.output
    record_directory = Path(snapshot_file_case.test_file.with_suffix(".capture").read_text())
    base_directory = Path(os.environ[CI_GATE_RECORD_DIR_ENV])
    assert record_directory.is_relative_to(base_directory)
    if existing_directory:
        assert base_directory == supplied_directory
    assert json.loads((record_directory / "probe.jsonl").read_text()) == {
        "metric": "train/grad_norm",
        "series": [[0, 1.5]],
    }
    assert record_directory.with_suffix(".merged.jsonl").is_file()


@pytest.mark.parametrize("golden_value", ["actual", "different"])
def test_successful_child_completes_attempt_and_checks_the_golden(
    snapshot_file_case: _SnapshotFileCase, golden_value: str
) -> None:
    """A real successful child completes its attempt even when the snapshot comparison fails."""
    snapshot_file_case.write_golden(value=golden_value)
    before = snapshot_file_case.golden.read_bytes()

    result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"])

    assert result.exit_code == (0 if golden_value == "actual" else -1), result.output
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {"test": str(snapshot_file_case.test_file), "completed": True}
    assert (attempt.parent / "records/record.json").is_file()
    assert snapshot_file_case.golden.read_bytes() == before


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_failed_or_timed_out_child_keeps_an_incomplete_attempt(
    snapshot_file_case: _SnapshotFileCase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    """Real child failure and timeout retain raw records without claiming successful completion."""
    snapshot_file_case.write_golden(value="actual")
    caplog.set_level(logging.INFO)
    monkeypatch.setenv(
        "FILE_RUN_TEST_EXIT" if failure == "exit" else "FILE_RUN_TEST_SLEEP", "7" if failure == "exit" else "60"
    )

    result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "2"])

    assert result.exit_code == -1, result.output
    assert ("returned exit code 7" if failure == "exit" else "after 2 seconds") in caplog.text
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {"test": str(snapshot_file_case.test_file), "completed": False}
    assert (attempt.parent / "records/record.json").is_file()
