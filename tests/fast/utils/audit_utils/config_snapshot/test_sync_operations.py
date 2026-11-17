import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch


from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt, ConfigSnapshotVariant
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.config_snapshot.sync import download, sync
from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY


class TestSnapshotArtifactDownload:
    def test_explicit_repository_and_empty_destination_reach_the_github_cli(self, tmp_path: Path) -> None:
        """Download raw record artifacts from the requested repository into the requested directory."""
        destination = tmp_path / "download with spaces"
        with patch("miles.utils.audit_utils.config_snapshot.sync.subprocess.run") as run:
            download(run_id="123456", directory=destination, repo="owner/private-repo")

        run.assert_called_once_with(
            [
                "gh",
                "run",
                "download",
                "123456",
                "--repo",
                "owner/private-repo",
                "--pattern",
                "snapshot-records-*",
                "--dir",
                str(destination),
            ],
            check=True,
        )

    def test_nonempty_destination_is_rejected_before_a_download(self, tmp_path: Path) -> None:
        """Existing downloaded evidence cannot be mixed with another workflow run."""
        existing = tmp_path / "preserved.json"
        existing.write_text("preserve evidence")
        with patch("miles.utils.audit_utils.config_snapshot.sync.subprocess.run") as run:
            with pytest.raises(ValueError, match="must be empty"):
                download(run_id="123456", directory=tmp_path, repo="owner/private-repo")

        run.assert_not_called()
        assert existing.read_text() == "preserve evidence"

    def test_github_cli_failure_is_not_reported_as_a_success(self, tmp_path: Path) -> None:
        """A failed artifact download remains visible to callers."""
        failure = subprocess.CalledProcessError(returncode=7, cmd=["gh", "run", "download"])
        with patch("miles.utils.audit_utils.config_snapshot.sync.subprocess.run", side_effect=failure):
            with pytest.raises(subprocess.CalledProcessError) as error:
                download(run_id="123456", directory=tmp_path, repo="owner/private-repo")

        assert error.value is failure


class TestSnapshotSyncWrites:
    def test_completed_records_replace_only_the_matching_variant_golden(
        self, tmp_path: Path, make_record: Callable
    ) -> None:
        """Promote completed raw records without modifying another test's existing golden."""
        attempt = tmp_path / "download" / "attempt"
        attempt.mkdir(parents=True)
        (attempt / "attempt.json").write_text(
            ConfigSnapshotTestAttempt(
                test="tests/e2e/test_training.py",
                variant=ConfigSnapshotVariant(hardware="h200", wandb_available=False),
                completed=True,
            ).model_dump_json()
        )
        ConfigSnapshotStorage(directory=attempt / "records").write(make_record(config={"value": "updated"}))
        root = tmp_path / "repo"
        target = root / SNAPSHOT_DIRECTORY / "runtime_config/variants/h200/no-wandb/tests/e2e/test_training.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("obsolete golden")
        unrelated = target.with_name("test_other.yaml")
        unrelated.write_text("preserve other golden")

        sync(directory=tmp_path / "download", repo_root=root)

        assert "value: updated" in target.read_text()
        assert "obsolete golden" not in target.read_text()
        assert unrelated.read_text() == "preserve other golden"
        assert set(root.rglob("*.yaml")) == {target, unrelated}
