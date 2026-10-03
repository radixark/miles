from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.generated_values import read_generated_values
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.test_utils.snapshot import dump_snapshot
from miles.utils.tracking_utils.wandb_utils import init_wandb_primary


class TestTrackingCapture:
    @pytest.mark.parametrize("run_id", ["explicit-one", "explicit-two"])
    def test_environment_run_ids_normalize_without_changing_the_active_run(
        self, wandb_capture_args: Namespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_id: str
    ) -> None:
        """Snapshot normalization preserves the actual environment-selected W&B run."""
        monkeypatch.setenv("WANDB_RUN_ID", run_id)
        init_wandb_primary(wandb_capture_args)
        ConfigSnapshotDumper.dump(stage="train_first_step", config={"args": wandb_capture_args})
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert wandb_capture_args.wandb_run_id == run_id
        assert read_generated_values() == []
        snapshot = dump_snapshot(ConfigSnapshotConverter.convert(records))
        assert run_id not in snapshot
        assert "$WANDB_RUN_ID_0000" in snapshot

    @pytest.mark.parametrize("preassigned", [False, True])
    def test_primary_captures_the_initialized_run_without_provenance(
        self, wandb_capture_args: Namespace, tmp_path: Path, preassigned: bool
    ) -> None:
        """Primary capture records both fresh and explicitly selected runs without ID provenance."""
        if preassigned:
            wandb_capture_args.wandb_run_id = "automatic-id"
        init_wandb_primary(wandb_capture_args)
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        tracking = [record for record in records if record.point.stage == "tracking_config"]
        assert len(tracking) == 1
        assert tracking[0].generated_values == []
        assert tracking[0].config["args"]["wandb_run_id"] == "automatic-id"
        assert (
            tracking[0].context == next(record for record in records if record.point.stage == "process_config").context
        )
