import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path
from types import ModuleType

from tests.fast.backends.megatron_utils.test_shared_ppo_lifecycle import actor_module
from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration, _cpu_worker_environment

from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage


__all__ = ["actor_module"]


class TestFirstMegatronTrainingSnapshot:
    @pytest.mark.parametrize("role", ["actor", "critic"])
    def test_each_constructed_role_records_its_first_training_configuration_only_once(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
    ) -> None:
        """Worker construction arms one initial training capture for the actor and critic independently."""
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=tmp_path)
        _cpu_worker_environment(monkeypatch)
        args = Namespace(
            **(
                vars(settings)
                | {"trainer_model_id": None, "object_store_backend": "ray", "worker_comm_backend": "ray", "lr": 0.1}
            )
        )
        worker = actor_module.MegatronTrainRayActor(args=args, world_size=1, rank=0, role=role, cell_index=0)
        worker.role = role

        worker._log_first_train_config()
        worker.args.lr = 0.2
        worker._log_first_train_config()

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert sorted(record.point.stage for record in records) == ["process_config", "train_first_step"]
        record = next(record for record in records if record.point.stage == "train_first_step")
        process = next(record for record in records if record.point.stage == "process_config")
        assert record.point.index == 0
        assert record.context == process.context
        assert record.config["args"]["lr"] == 0.1
        assert record.config["role"] == role
