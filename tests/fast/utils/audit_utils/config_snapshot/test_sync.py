from collections.abc import Callable
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.config_snapshot.sync import _collect_snapshots


def test_completed_unit_test_without_records_does_not_block_training_snapshot(
    tmp_path: Path, make_record: Callable
) -> None:
    """An uploaded archive can omit an empty records directory without losing another test's snapshot."""
    raw = tmp_path / "raw"
    empty = raw / "unit" / "attempt"
    populated = raw / "training" / "attempt"
    _write_attempt(directory=empty, test="tests/fast-gpu/test_unit.py")
    _write_attempt(directory=populated, test="tests/e2e/test_training.py")
    ConfigSnapshotStorage(directory=populated / "records").write(make_record(config={"value": "retained"}))
    _write_bases(repo_root=tmp_path)

    snapshots = _collect_snapshots(directory=raw, repo_root=tmp_path)

    assert list(snapshots) == [tmp_path / "tests/snapshots/runtime_config/tests/e2e/test_training.yaml"]
    assert "value: retained" in next(iter(snapshots.values()))
    assert (empty / "attempt.json").is_file()
    assert not (empty / "records").exists()


def test_empty_and_populated_completed_attempts_for_one_test_still_disagree(
    tmp_path: Path, make_record: Callable
) -> None:
    """An empty completed attempt cannot be filtered out to hide a same-test coverage difference."""
    raw = tmp_path / "raw"
    empty = raw / "test" / "empty"
    populated = raw / "test" / "populated"
    for directory in (empty, populated):
        _write_attempt(directory=directory, test="tests/e2e/test_training.py")
    ConfigSnapshotStorage(directory=populated / "records").write(make_record(config={"value": "retained"}))
    _write_bases(repo_root=tmp_path)

    with pytest.raises(ValueError, match="Completed attempts disagree"):
        _collect_snapshots(directory=raw, repo_root=tmp_path)


def _write_attempt(*, directory: Path, test: str) -> None:
    directory.mkdir(parents=True)
    (directory / "attempt.json").write_text(ConfigSnapshotTestAttempt(test=test, completed=True).model_dump_json())


def _write_bases(*, repo_root: Path) -> None:
    path = repo_root / "tests/snapshots/runtime_config/base.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("templates:\n  default: {}\n")
