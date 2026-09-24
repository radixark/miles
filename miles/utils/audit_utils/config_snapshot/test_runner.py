import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.snapshot import (
    SNAPSHOT_RECORD_DIR_ENV_VAR,
    SNAPSHOT_UPDATE_ENV_VAR,
    assert_matches_snapshot,
    dump_snapshot,
)

logger = logging.getLogger(__name__)


class ConfigSnapshotMismatch(AssertionError):
    pass


@dataclass(frozen=True)
class ConfigSnapshotTestRunner:
    test: str
    directory: Path
    storage: ConfigSnapshotStorage
    golden: Path

    @property
    def record_directory(self) -> Path:
        return self.storage.directory

    @classmethod
    def create(cls, *, test: str, repo_root: Path) -> "ConfigSnapshotTestRunner":
        if not (record_directory := os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR)):
            raise ValueError(f"{SNAPSHOT_RECORD_DIR_ENV_VAR} is required for configuration snapshots")
        root = Path(record_directory).resolve()
        directory = root / Path(test).stem / uuid.uuid4().hex
        attempt = cls(
            test=test,
            directory=directory,
            storage=ConfigSnapshotStorage(directory=directory / "records"),
            golden=cls.golden_path(test=test, repo_root=repo_root),
        )
        attempt.record_directory.mkdir(parents=True)
        attempt._write_status(completed=False)
        return attempt

    def finish(self, *, returncode: int) -> None:
        self._write_status(completed=returncode == 0)
        if returncode != 0:
            return
        try:
            case = ConfigSnapshotConverter.convert(self.storage.read())
            if not case.processes:
                return
            assert_matches_snapshot(
                snapshot=self.golden,
                actual=dump_snapshot(case),
                subject=f"runtime configuration; raw dumps: {self.record_directory}",
                update=bool(os.environ.get(SNAPSHOT_UPDATE_ENV_VAR)),
            )
        except Exception as error:
            logger.exception("Snapshot analysis failed; raw dumps: %s", self.record_directory)
            raise ConfigSnapshotMismatch(str(error)) from error

    @staticmethod
    def golden_path(*, test: str, repo_root: Path) -> Path:
        root = repo_root.resolve()
        relative = Path(test)
        if relative.is_absolute():
            relative = relative.relative_to(root)
        if ".." in relative.parts:
            raise ValueError(f"Test path must stay within the repository: {test}")
        target = (root / "tests/snapshots/runtime_config" / relative.with_suffix(".yaml")).resolve()
        if not target.is_relative_to(root / "tests/snapshots/runtime_config"):
            raise ValueError(f"Snapshot target must stay within the snapshot directory: {target}")
        return target

    def _write_status(self, *, completed: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path=self.directory / "attempt.json",
            text=ConfigSnapshotTestAttempt(test=self.test, completed=completed).model_dump_json(),
        )
