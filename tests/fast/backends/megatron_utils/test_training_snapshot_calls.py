import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path
from types import ModuleType

from tests.fast.backends.megatron_utils.test_shared_ppo_lifecycle import (
    _actor_worker,
    _critic_worker,
    _FakeRay,
    _noop_timer,
    _patch_shared_train_helpers,
    actor_module,
)
from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration, _stop_training_loop

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity

__all__ = ["actor_module"]


class TestMegatronTrainingCaptureCalls:
    @pytest.mark.parametrize("role", ["actor", "critic"])
    def test_each_backend_training_path_captures_configuration_before_the_optimizer_loop(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
    ) -> None:
        """Actor and critic training must invoke capture before handing work to the optimizer loop."""
        _patch_shared_train_helpers(actor_module, monkeypatch, _FakeRay())
        monkeypatch.setattr(actor_module, "forward_only", lambda *args, **kwargs: {})
        monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)
        monkeypatch.setattr(actor_module, "log_rollout_data", lambda *args, **kwargs: None)
        monkeypatch.setattr(actor_module, "log_train_advantage_computation_event", lambda *args, **kwargs: None)
        monkeypatch.setattr(actor_module, "train", _stop_training_loop)
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=tmp_path)
        ConfigSnapshotDumper.configure(
            args=settings, source=TrainProcessIdentity(component=role, cell_index=0, rank_within_cell=0)
        )
        worker = _critic_worker(actor_module) if role == "critic" else _actor_worker(actor_module)
        worker._config_snapshot_train_recorded = False

        with pytest.raises(RuntimeError, match="training SDK stopped"):
            if role == "critic":
                worker._train_critic(rollout_id=0, rollout_data={})
            else:
                worker.args.use_critic = False
                worker._train_actor(0, {"tokens": []}, None, witness_info=None, attempt=0)

        [record] = ConfigSnapshotStorage(directory=tmp_path).read()
        assert record.point.stage == "train_first_step"
        assert record.point.index == 0
        assert record.config["role"] == role
        assert record.config["args"]["backend"]["global_batch_size"] == 1
