import re
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci
from tests.ci.test.snapshot_artifact_fixtures import _artifact_name

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_ROOT = Path(__file__).resolve().parents[3]


class TestSnapshotRecordArtifacts:
    @pytest.mark.parametrize("workflow_name", ["_run-ci.yml", "_run-ci-rocm.yml", "_run-cpu-ci.yml"])
    def test_failed_test_commands_keep_their_recordings_and_empty_runs_remain_valid(
        self, workflow_name: str, tmp_path: Path
    ) -> None:
        """Every test workflow uploads its recording directory even after failures."""
        workflow = yaml.safe_load((_ROOT / ".github/workflows" / workflow_name).read_text())
        recording_jobs = [
            job
            for job in workflow["jobs"].values()
            if any("MILES_SNAPSHOT_RECORD_DIR" in step.get("env", {}) for step in job.get("steps", []))
        ]
        assert recording_jobs

        for index, job in enumerate(recording_jobs):
            steps = job["steps"]
            recorder = next(step for step in steps if "MILES_SNAPSHOT_RECORD_DIR" in step.get("env", {}))
            directory = recorder["env"]["MILES_SNAPSHOT_RECORD_DIR"]
            uploads = [
                step
                for step in steps
                if step.get("uses", "").startswith("actions/upload-artifact@") and step["with"]["path"] == directory
            ]
            assert len(uploads) == 1
            upload = uploads[0]
            assert upload["if"] == "always()"
            assert upload["with"]["if-no-files-found"] == "ignore"

            reference = re.fullmatch(r"\$\{\{\s*steps\.([\w-]+)\.outputs\.name\s*\}\}", upload["with"]["name"])
            assert reference is not None
            name_step = next(step for step in steps if step.get("id") == reference.group(1))
            assert name_step["if"] == "always()"
            assert name_step["env"]["EXECUTE_COMMAND"] == recorder["run"]

            commands = [
                "pytest --suite stage-a-cpu --auto-partition-id 0",
                "pytest --suite stage-a-cpu --auto-partition-id 1",
                "pytest --suite stage-a-cpu --auto-partition-id 0 -k other",
                "pytest tests/fast",
            ]
            names = [
                _artifact_name(script=name_step["run"], command=command, output=tmp_path / f"{index}-{case}.txt")
                for case, command in enumerate(commands)
            ]
            assert len(set(names)) == len(commands)
            assert re.fullmatch(r"snapshot-records-stage-a-cpu-p0-[0-9a-f]{8}", names[0])
            assert re.fullmatch(r"snapshot-records-command-[0-9a-f]{8}", names[3])
            assert (
                _artifact_name(script=name_step["run"], command=commands[0], output=tmp_path / f"{index}-repeat.txt")
                == names[0]
            )
