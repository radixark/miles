import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

from tests.fast.fixtures.args_fixtures import make_trainer_args
from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration, _stop_checkpoint_load

from miles.backends.megatron_utils import checkpoint
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


class TestMegatronCheckpointSnapshot:
    @pytest.mark.parametrize("local_manager", [False, True])
    def test_checkpoint_configuration_is_captured_before_native_or_hf_loading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, local_manager: bool
    ) -> None:
        """Both checkpoint loaders expose their selected path and runtime configuration before SDK access."""
        records = tmp_path / "records"
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=records)
        ConfigSnapshotDumper.configure(
            args=settings, source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        )
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "config.json").write_text("{}")
        args = make_trainer_args(load=str(weights), spec=None)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", _stop_checkpoint_load)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", _stop_checkpoint_load)

        with pytest.raises(RuntimeError, match="checkpoint SDK stopped"):
            checkpoint.load_checkpoint(
                ddp_model=[],
                optimizer=None,
                opt_param_scheduler=None,
                checkpointing_context={"local_checkpoint_manager": object()} if local_manager else None,
                skip_load_to_model_and_opt=False,
                args=args,
            )

        [record] = ConfigSnapshotStorage(directory=records).read()
        assert record.point.stage == "checkpoint_load"
        assert record.config["load"] == str(weights)
        assert record.config["local_checkpoint_manager"] is local_manager
        assert record.config["args"]["backend"]["load"] == str(weights)
