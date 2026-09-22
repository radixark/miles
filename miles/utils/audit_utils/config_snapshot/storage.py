import uuid
from dataclasses import dataclass
from pathlib import Path

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotRecord
from miles.utils.file_utils import atomic_write_text


@dataclass(kw_only=True)
class ConfigSnapshotStorage:
    directory: Path

    def write(self, record: ConfigSnapshotRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=self.directory / f"{uuid.uuid4().hex}.json", text=record.model_dump_json())

    def read(self) -> list[ConfigSnapshotRecord]:
        return [
            ConfigSnapshotRecord.model_validate_json(path.read_text())
            for path in sorted(self.directory.iterdir())
            if path.suffix == ".json"
        ]
