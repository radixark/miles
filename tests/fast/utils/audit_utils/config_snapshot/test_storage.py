from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage


def test_malformed_record_is_not_treated_as_an_empty_directory(tmp_path: Path) -> None:
    """Corrupt JSON remains a validation failure after missing archive directories become empty records."""
    (tmp_path / "record.json").write_text("not-json")

    with pytest.raises(ValidationError):
        ConfigSnapshotStorage(directory=tmp_path).read()


def test_non_directory_storage_path_still_raises(tmp_path: Path) -> None:
    """A file occupying the records directory is not a valid empty snapshot."""
    path = tmp_path / "records"
    path.write_text("not-a-directory")

    with pytest.raises(NotADirectoryError):
        ConfigSnapshotStorage(directory=path).read()
