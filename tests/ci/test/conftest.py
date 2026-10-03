from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.ci import ci_utils

from miles.utils.audit_utils.config_snapshot.compact import ConfigSnapshotBases
from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotContext,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotTestRunner
from miles.utils.audit_utils.config_snapshot.serialization import dump_config_snapshot
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR, dump_snapshot


@dataclass(frozen=True)
class _SnapshotFileCase:
    test_file: Path
    record_root: Path
    golden: Path
    bases: Path
    record: ConfigSnapshotRecord

    def write_golden(self, *, value: str) -> None:
        record = self.record.model_copy(update={"config": {"args": {"value": value}}})
        self.golden.parent.mkdir(parents=True, exist_ok=True)
        bases = ConfigSnapshotBases(templates={"default": {}})
        self.bases.write_text(dump_snapshot(bases))
        self.golden.write_text(dump_config_snapshot(ConfigSnapshotConverter.convert([record]), bases=bases))


@pytest.fixture
def snapshot_file_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SnapshotFileCase:
    relative = Path("tests/e2e/test_snapshot_probe.py")
    script = tmp_path / relative
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json\n"
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "records = Path(os.environ['MILES_SNAPSHOT_RECORD_DIR'])\n"
        "records.mkdir(parents=True, exist_ok=True)\n"
        "(records / 'record.json').write_text(Path(__file__).with_suffix('.json').read_text())\n"
        "metric_dir = Path(os.environ['MILES_CI_GATE_RECORD_DIR'])\n"
        "Path(__file__).with_suffix('.capture').write_text(str(metric_dir))\n"
        "(metric_dir / 'probe.jsonl').write_text(json.dumps({'metric': 'train/grad_norm', 'series': [[0, 1.5]]}) + '\\n')\n"
        "time.sleep(float(os.environ.get('FILE_RUN_TEST_SLEEP', '0')))\n"
        "raise SystemExit(int(os.environ.get('FILE_RUN_TEST_EXIT', '0')))\n"
    )
    record = ConfigSnapshotRecord(
        context=ConfigSnapshotContext(
            name="tests/e2e/test_snapshot_probe/run-0000",
            deploy_component="all",
            deploy_instance_id="default",
            source=SimpleProcessIdentity(component="main"),
            run_uuid="test-run",
            capture_id="test-capture",
        ),
        point=ConfigSnapshotPoint(stage="process_config", index=0),
        config={"args": {"value": "actual"}},
    )
    script.with_suffix(".json").write_text(record.model_dump_json())
    record_root = tmp_path / "record-root"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci_utils, "__file__", str(tmp_path / "tests/ci/ci_utils.py"))
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(record_root))
    monkeypatch.setenv("CI", "false")
    monkeypatch.setenv(ci_utils.CI_GATE_RECORD_DIR_ENV, "")
    for name in (
        SNAPSHOT_UPDATE_ENV_VAR,
        "FILE_RUN_TEST_SLEEP",
        "FILE_RUN_TEST_EXIT",
        ci_utils.CI_GATE_RECORD_DIR_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return _SnapshotFileCase(
        test_file=relative,
        record_root=record_root,
        golden=ConfigSnapshotTestRunner.golden_path(test=str(relative), repo_root=tmp_path),
        bases=tmp_path / "tests/snapshots/runtime_config/base.yaml",
        record=record,
    )
