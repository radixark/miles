import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotContext,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


class TestRawRecordStorage:
    def test_equal_stage_indices_from_distinct_ranks_remain_separate_and_ignore_non_json_files(
        self, tmp_path: Path
    ) -> None:
        """Raw storage preserves every rank's record without treating incomplete side files as captures."""
        records = [
            ConfigSnapshotRecord(
                context=ConfigSnapshotContext(
                    name="storage-probe",
                    deploy_component="all",
                    deploy_instance_id="default",
                    source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=rank),
                    run_uuid="storage-run",
                ),
                point=ConfigSnapshotPoint(stage="process_config", index=0),
                config={"rank": rank, "values": [None, False, rank]},
            )
            for rank in range(2)
        ]
        storage = ConfigSnapshotStorage(directory=tmp_path / "records")
        for record in records:
            storage.write(record)
        side_file = storage.directory / "unfinished.tmp"
        side_file.write_text("not-json")

        restored = sorted(storage.read(), key=lambda record: record.context.source.rank_within_cell)

        assert restored == records
        assert len(list(storage.directory.glob("*.json"))) == 2
        assert side_file.read_text() == "not-json"
