import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

from tests.fast.fixtures.args_fixtures import make_trainer_args
from tests.fast.fixtures.config_snapshot_fixtures import _capture_configuration, _stop_checkpoint_load
from transformers import AutoModelForCausalLM

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


class TestFsdpCheckpointSnapshot:
    def test_hf_model_construction_captures_its_format_and_path_before_loading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """HF model startup records its selected checkpoint before constructing model parameters."""
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=tmp_path)
        ConfigSnapshotDumper.configure(
            args=settings, source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        )
        worker = object.__new__(FSDPTrainRayActor)
        worker.args = make_trainer_args(train_backend="fsdp", attn_implementation="eager")
        worker.hf_config = Namespace(model_type="snapshot-test-model")
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _stop_checkpoint_load)

        with pytest.raises(RuntimeError, match="checkpoint SDK stopped"):
            worker._build_model_with_attn_bridge(checkpoint_path="/selected/hf-model", init_context=nullcontext)

        [record] = ConfigSnapshotStorage(directory=tmp_path).read()
        assert record.point.stage == "checkpoint_load"
        assert record.config["load"] == "/selected/hf-model"
        assert record.config["format"] == "hf"
        assert record.config["args"]["backend"]["attn_implementation"] == "eager"

    def test_native_checkpoint_load_captures_the_resolved_iteration_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Native FSDP loading records the concrete iteration rather than only its checkpoint root."""
        records = tmp_path / "records"
        settings = _capture_configuration(monkeypatch=monkeypatch, directory=records)
        ConfigSnapshotDumper.configure(
            args=settings, source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        )
        root = tmp_path / "checkpoints"
        resolved = root / "iter_0000002"
        (resolved / "model").mkdir(parents=True)
        args = make_trainer_args(train_backend="fsdp", load=str(root), ckpt_step=2)
        worker = Namespace(args=args, model=object())
        monkeypatch.setattr(checkpoint.dcp, "load", _stop_checkpoint_load)

        assert checkpoint.load(worker) is None

        [record] = ConfigSnapshotStorage(directory=records).read()
        assert record.point.stage == "checkpoint_load"
        assert record.config["load"] == str(resolved)
        assert record.config["format"] == "fsdp"
        assert record.config["args"]["backend"]["load"] == str(root)
