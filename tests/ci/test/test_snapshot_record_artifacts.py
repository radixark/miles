import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_ROOT = Path(__file__).resolve().parents[3]


class TestSnapshotRecordArtifacts:
    def test_workflows_do_not_enable_snapshot_recording_or_uploads(self) -> None:
        """All CI workflows leave snapshot recording and comparison disabled."""
        for path in (_ROOT / ".github/workflows").glob("*.yml"):
            contents = path.read_text()
            workflow = yaml.safe_load(contents)
            assert "MILES_SNAPSHOT_RECORD_DIR" not in contents, path
            assert "MILES_UPDATE_SNAPSHOTS" not in contents, path
            for job in workflow.get("jobs", {}).values():
                for step in job.get("steps", []):
                    assert step.get("id") != "snapshot-record-name", path
                    if step.get("uses", "").startswith("actions/upload-artifact@"):
                        assert "snapshot-records" not in str(step["with"]), path
