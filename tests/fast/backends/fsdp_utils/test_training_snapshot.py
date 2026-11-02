import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path


from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestFirstTrainingSnapshot:
    def test_repeated_training_records_only_the_initial_configuration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The first FSDP training call records configuration once before later state changes."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        ConfigSnapshotDumper.configure(
            args=Namespace(
                ci_disable_config_snapshot=False,
                ci_test=True,
                config_snapshot_name="fsdp",
                deploy_component="all",
                deploy_instance_id=None,
                run_uuid="test-run",
            ),
            source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0),
        )
        actor = object.__new__(FSDPTrainRayActor)
        actor.args = Namespace(lr=0.1)
        actor.role = "actor"
        actor._config_snapshot_train_recorded = False
        actor._before_train()
        actor.args.lr = 0.2
        actor._before_train()

        [record] = ConfigSnapshotStorage(directory=tmp_path).read()
        assert record.point.stage == "train_first_step"
        assert record.point.index == 0
        assert record.config == {"args": {"lr": 0.1}, "role": "actor"}
