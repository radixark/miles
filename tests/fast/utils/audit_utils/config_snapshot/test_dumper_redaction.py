import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.env_report.redaction import redact_arg


class TestCapturedConfigurationRedaction:
    def test_captured_credentials_are_redacted_without_mutating_live_configuration(
        self, capture_args: Namespace, tmp_path: Path
    ) -> None:
        """Raw snapshot capture cannot persist a credential or change the configuration used by training."""
        config = {"args": {"wandb_key": "snapshot-secret", "clip_grad": 0.75}, "public": "visible"}

        ConfigSnapshotDumper.dump(stage="train_first_step", config=config)
        record = next(
            record
            for record in ConfigSnapshotStorage(directory=tmp_path).read()
            if record.point.stage == "train_first_step"
        )

        assert record.config == {
            "args": {"wandb_key": redact_arg("wandb_key", "snapshot-secret"), "clip_grad": 0.75},
            "public": "visible",
        }
        assert "snapshot-secret" not in record.model_dump_json()
        assert config == {"args": {"wandb_key": "snapshot-secret", "clip_grad": 0.75}, "public": "visible"}
