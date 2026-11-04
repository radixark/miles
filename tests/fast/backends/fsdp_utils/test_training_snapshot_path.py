import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration, _empty_rollout_data
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.training_utils.torch_native import actor as native_actor
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.ft_utils.heartbeat_utils import SimpleHeartbeat
from miles.utils.timer import Timer


class TestFsdpTrainingCapturePath:
    def test_public_training_calls_capture_the_initial_configuration_once_before_reading_rollout_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Native training must invoke the FSDP capture hook before its rollout-data adapter."""
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=tmp_path)
        ConfigSnapshotDumper.configure(
            args=settings, source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        )
        worker = object.__new__(FSDPTrainRayActor)
        worker.args = Namespace(lr=0.1, offload_train=False, debug_rollout_only=True)
        worker.role = "actor"
        worker._config_snapshot_train_recorded = False
        worker._heartbeat = SimpleHeartbeat()
        parallel_config = make_train_parallel_config()
        monkeypatch.setattr(
            native_actor,
            "get_parallel_state",
            lambda: SimpleNamespace(train_parallel_config=lambda **kwargs: parallel_config),
        )
        monkeypatch.setattr(native_actor, "get_rollout_data", _empty_rollout_data)
        monkeypatch.setattr(Timer(), "start_time", {})
        monkeypatch.setattr(Timer(), "timers", {})
        monkeypatch.setattr(Timer(), "event_sinks", [])
        Timer().start("train_wait")

        worker.train(rollout_id=0, rollout_data_ref=object())
        worker.args.lr = 0.2
        worker.train(rollout_id=1, rollout_data_ref=object())

        [record] = ConfigSnapshotStorage(directory=tmp_path).read()
        assert record.point.stage == "train_first_step"
        assert record.config["args"]["lr"] == 0.1
        assert record.config["role"] == "actor"
