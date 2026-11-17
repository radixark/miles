import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import json
import logging
import os
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci
from tests.ci.ci_utils import CI_GATE_RECORD_DIR_ENV, TestFile, run_unittest_files
from tests.ci.run_file import app
from tests.ci.test.conftest import _SnapshotFileCase
from typer.testing import CliRunner

from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR

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

    result = CliRunner().invoke(
        app,
        ["--suite", "stage-c-2-gpu-h200", "--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"],
    )

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

    result = CliRunner().invoke(
        app,
        ["--suite", "stage-c-2-gpu-h200", "--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"],
    )

    assert result.exit_code == (0 if golden_value == "actual" else -1), result.output
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {
        "test": str(snapshot_file_case.test_file),
        "completed": True,
        "variant": {"hardware": "h200", "wandb_available": False},
    }
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

    result = CliRunner().invoke(
        app,
        ["--suite", "stage-c-2-gpu-h200", "--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "2"],
    )

    assert result.exit_code == -1, result.output
    assert ("returned exit code 7" if failure == "exit" else "after 2 seconds") in caplog.text
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {
        "test": str(snapshot_file_case.test_file),
        "completed": False,
        "variant": {"hardware": "h200", "wandb_available": False},
    }
    assert (attempt.parent / "records/record.json").is_file()


@pytest.mark.parametrize("record_directory", [None, ""])
@pytest.mark.parametrize("standalone", [False, True])
def test_child_without_capture_runs_without_injecting_a_record_directory(
    snapshot_file_case: _SnapshotFileCase,
    monkeypatch: pytest.MonkeyPatch,
    record_directory: str | None,
    standalone: bool,
) -> None:
    """Standalone and suite runners preserve ordinary child execution when snapshot capture is disabled."""
    snapshot_file_case.test_file.write_text(
        "import os\nfrom pathlib import Path\n"
        "assert not os.environ.get('MILES_SNAPSHOT_RECORD_DIR')\n"
        "Path(__file__).with_suffix('.ran').touch()\n"
    )
    if record_directory is None:
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR)
    else:
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, record_directory)

    if standalone:
        result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"])
        assert result.exit_code == 0, result.output
    else:
        assert (
            run_unittest_files(
                [TestFile(name=str(snapshot_file_case.test_file), estimated_time=0)],
                timeout_per_file=10,
                enable_retry=False,
                continue_on_error=False,
            )
            == 0
        )

    assert snapshot_file_case.test_file.with_suffix(".ran").exists()
    assert not snapshot_file_case.record_root.exists()
    assert os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR) == record_directory


def test_enabled_capture_rejects_a_missing_golden(snapshot_file_case: _SnapshotFileCase) -> None:
    """Explicitly enabled recording still requires a runtime baseline for captured configuration."""
    result = CliRunner().invoke(
        app,
        ["--suite", "stage-c-2-gpu-h200", "--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"],
    )

    assert result.exit_code == -1, result.output
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text())["completed"] is True
    assert not snapshot_file_case.golden.exists()


@pytest.mark.parametrize("suite", [None, "unknown-h200"])
def test_capture_without_a_known_execution_suite_cannot_run_the_child(
    snapshot_file_case: _SnapshotFileCase, suite: str | None
) -> None:
    """Recording cannot silently attribute a standalone execution to a guessed hardware."""
    arguments = ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"]
    if suite is not None:
        arguments += ["--suite", suite]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert not snapshot_file_case.test_file.with_suffix(".capture").exists()


@pytest.mark.parametrize("available", [False, True])
def test_actual_suite_and_credential_availability_are_frozen_without_secrets(
    snapshot_file_case: _SnapshotFileCase, monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    """Single-file capture uses the selected suite and stores only credential availability."""
    monkeypatch.setenv("MILES_UPDATE_SNAPSHOTS", "1")
    if available:
        monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret-not-for-recording")
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        ["--suite", "stage-c-4-gpu-b200", "--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"],
    )

    assert result.exit_code == 0, result.output
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text())["variant"] == {"hardware": "b200", "wandb_available": available}
    assert "synthetic-secret" not in attempt.read_text()
    assert not snapshot_file_case.golden.exists()
