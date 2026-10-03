from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from miles.utils.audit_utils.config_snapshot.compact import CompactConfigSnapshotCase
from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotMismatch, ConfigSnapshotTestRunner
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR


def test_snapshot_update_writes_overrides_without_modifying_shared_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_record: Callable
) -> None:
    """Snapshot updates retain the manually maintained template and record changed values in the case."""
    bases = tmp_path / "tests/snapshots/runtime_config/base.yaml"
    bases.parent.mkdir(parents=True)
    original = "templates:\n  default:\n    args:\n      value: old\n"
    bases.write_text(original)
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
    monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
    runner = ConfigSnapshotTestRunner.create(test="tests/e2e/test_training.py", repo_root=tmp_path)
    runner.storage.write(make_record(config={"value": "new"}))

    runner.finish(returncode=0)

    case = CompactConfigSnapshotCase.model_validate(yaml.safe_load(runner.golden.read_text()))
    process = next(iter(case.processes.values()))
    assert process.template == "default"
    assert process.overrides.model_dump() == {"set": {"/args/value": "new"}, "remove": []}
    assert bases.read_text() == original
    monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR)
    bases.write_text("templates:\n  default:\n    args:\n      value: new\n")
    with pytest.raises(ConfigSnapshotMismatch, match="does not match"):
        runner.finish(returncode=0)
