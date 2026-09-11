import pytest
import torch

from miles.backends.training_utils.model_companion import (
    ModelCompanion,
    ModelCompanionInstallationUtils,
    ModelCompanionSampleConsumptionUtils,
    ModelCompanionWeightVersionUtils,
)
from miles.utils.types import SampleLineage


def _identity(source_sample_index: int, output_index: int, output_count: int) -> SampleLineage:
    return SampleLineage(
        source_sample_index=source_sample_index,
        output_index=output_index,
        output_count=output_count,
    )


class TestModelCompanion:
    def test_state_is_non_trainable_cpu_parameters(self) -> None:
        """Normal parameter enumeration includes the witness without optimizer gradients."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))

        assert set(dict(witness.named_parameters())) == {"sample_consumptions", "weight_version"}
        assert list(witness.buffers()) == []
        assert all(
            parameter.device.type == "cpu" and not parameter.requires_grad for parameter in witness.parameters()
        )

    def test_install_uses_the_intra_cell_dp_replica_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FT cells describe the same checkpoint shard despite different alive ranks."""
        parallel = type(
            "Parallel",
            (),
            {
                "pp": type("Group", (), {"rank": 2})(),
                "tp": type("Group", (), {"rank": 3})(),
                "cp": type("Group", (), {"rank": 4})(),
                "intra_dp": type("Group", (), {"rank": 5})(),
                "effective_dp": type("Group", (), {"rank": 6})(),
            },
        )()
        monkeypatch.setattr(
            "miles.backends.training_utils.parallel.get_parallel_state",
            lambda: parallel,
        )
        model = torch.nn.Module()

        ModelCompanionInstallationUtils.install(model, chunk_index=1)

        assert model.model_companion.replica_id == (3, 4, 5)

    def test_hide_excludes_witness_from_pretrained_weight_load(self) -> None:
        """Pretrained weights can load while the training-only module is hidden."""
        model = torch.nn.Module()
        model.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))

        with ModelCompanionInstallationUtils.hide([model]):
            assert "model_companion" not in dict(model.named_children())

        assert "model_companion" in dict(model.named_children())

    def test_counts_every_occurrence(self) -> None:
        """Repeated sample identities remain visible as repeated consumption."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))

        witness.record_sample_consumptions([_identity(7, 0, 1), _identity(8, 0, 1), _identity(7, 0, 1)])

        assert witness.snapshot_sample_consumptions(is_skipped=False) == {_identity(7, 0, 1): 2, _identity(8, 0, 1): 1}

    def test_state_dict_round_trip_resizes_and_replaces_parameters(self) -> None:
        """Checkpoint restore resizes dynamic parameters and replaces both outcomes."""
        source = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        source.record_sample_consumptions([_identity(7, 0, 2), _identity(7, 1, 2), _identity(7, 0, 2)])
        source.record_sample_consumptions([_identity(8, 0, 1)], is_skipped=True)
        target = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        target.record_sample_consumptions([_identity(9, 0, 1)])
        target.record_sample_consumptions([_identity(10, 0, 1)], is_skipped=True)

        target.load_state_dict(source.state_dict())

        assert target.snapshot_sample_consumptions(is_skipped=False) == {_identity(7, 0, 2): 2, _identity(7, 1, 2): 1}
        assert target.snapshot_sample_consumptions(is_skipped=True) == {_identity(8, 0, 1): 1}

    def test_same_identity_keeps_trained_and_skipped_counts_separate(self) -> None:
        """The outcome flag distinguishes both counts within the same parameter."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        identity = _identity(7, 0, 1)
        witness.record_sample_consumptions([identity, identity])
        witness.record_sample_consumptions([identity], is_skipped=True)
        restored = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))

        restored.load_state_dict(witness.state_dict())

        assert restored.snapshot_sample_consumptions(is_skipped=False) == {identity: 2}
        assert restored.snapshot_sample_consumptions(is_skipped=True) == {identity: 1}
        assert restored.sample_consumptions.tolist() == [[7, 0, 1, 2, 0], [7, 0, 1, 1, 1]]

    def test_parent_device_move_keeps_parameters_on_cpu(self) -> None:
        """The CPU-only companion ignores parent model device transforms."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        witness.record_sample_consumptions([_identity(7, 0, 1)])

        model = torch.nn.Module()
        model.add_module("companion", witness)
        model.to(device="meta")

        assert witness.sample_consumptions.device.type == "cpu"

    def test_record_preserves_registered_parameter_identity(self) -> None:
        """Backuper references remain valid while the witness grows."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        parameter = witness.sample_consumptions

        witness.record_sample_consumptions([_identity(7, 0, 1)])
        witness.record_sample_consumptions([_identity(8, 0, 1)])

        assert witness.sample_consumptions is parameter
        assert parameter.shape == (2, 5)

    def test_checkpoint_requires_the_outcome_column(self) -> None:
        """A truncated row cannot silently discard its outcome flag."""
        witness = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))

        with pytest.raises(AssertionError):
            witness.load_state_dict({"sample_consumptions": torch.empty((0, 4), dtype=torch.int64)})

    def test_model_chunks_receive_identical_updates(self) -> None:
        """Virtual pipeline chunks retain identical witness mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))

        ModelCompanionSampleConsumptionUtils.record(chunks, [_identity(7, 0, 2), _identity(7, 1, 2)])
        ModelCompanionSampleConsumptionUtils.record(chunks, [_identity(8, 0, 1)], is_skipped=True)

        assert ModelCompanionSampleConsumptionUtils.snapshot(chunks, is_skipped=False) == {
            _identity(7, 0, 2): 1,
            _identity(7, 1, 2): 1,
        }
        assert ModelCompanionSampleConsumptionUtils.snapshot(chunks, is_skipped=True) == {_identity(8, 0, 1): 1}

    def test_snapshot_rejects_divergent_model_chunks(self) -> None:
        """A snapshot cannot silently choose between divergent chunk mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))
        chunks[0].model_companion.record_sample_consumptions([_identity(7, 0, 1)])
        chunks[1].model_companion.record_sample_consumptions([_identity(8, 0, 1)])

        with pytest.raises(AssertionError, match="chunks diverged"):
            ModelCompanionSampleConsumptionUtils.snapshot(chunks, is_skipped=False)


class TestModelWeightVersion:
    def test_successful_steps_advance_all_chunks_without_witness_rows(self) -> None:
        """Version progress does not depend on whether samples are recorded."""
        chunks = [
            ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)),
            ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)),
        ]
        assert ModelCompanionWeightVersionUtils.weight_version(chunks) == 0

        ModelCompanionWeightVersionUtils.bump_weight_version(chunks)
        ModelCompanionWeightVersionUtils.bump_weight_version(chunks)

        assert ModelCompanionWeightVersionUtils.weight_version(chunks) == 2
        assert all(chunk.snapshot_sample_consumptions(is_skipped=False) == {} for chunk in chunks)

    def test_checkpoint_restores_version_and_witness_as_one_state(self) -> None:
        """Loading older weights also rewinds their version and witness."""
        source = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        ModelCompanionWeightVersionUtils.bump_weight_version([source])
        source.record_sample_consumptions([_identity(7, 0, 1)])
        target = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        target.weight_version.fill_(90)
        target.record_sample_consumptions([_identity(8, 0, 1)])

        target.load_state_dict(source.state_dict())

        assert ModelCompanionWeightVersionUtils.weight_version([target]) == 1
        assert target.snapshot_sample_consumptions(is_skipped=False) == {_identity(7, 0, 1): 1}

    def test_version_is_a_cpu_tensor_preserved_by_model_transforms(self) -> None:
        """Device and dtype transforms leave the model version intact on CPU."""
        companion = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        original = companion.weight_version
        ModelCompanionWeightVersionUtils.bump_weight_version([companion])
        companion.to(device="meta", dtype=torch.float16)

        assert companion.weight_version is original
        assert original.device.type == "cpu"
        assert original.dtype == torch.int64
        assert original.item() == 1

    def test_divergent_chunks_fail_before_any_version_changes(self) -> None:
        """A version mismatch cannot be hidden by advancing every chunk."""
        chunks = [
            ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)),
            ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)),
        ]
        chunks[1].weight_version.fill_(2)

        with pytest.raises(AssertionError, match="versions diverged"):
            ModelCompanionWeightVersionUtils.bump_weight_version(chunks)

        assert [chunk.weight_version.item() for chunk in chunks] == [0, 2]

    def test_tensor_backuper_restores_version_with_actor_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Temporary model swaps restore the actor version and witness together."""
        from miles.utils.tensor_backper import TensorBackuper

        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        companion = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        companion.weight_version.fill_(7)
        companion.record_sample_consumptions([_identity(3, 0, 1)])
        backuper = TensorBackuper.create(
            lambda: ((f"model_companion.{name}", parameter) for name, parameter in companion.named_parameters())
        )
        backuper.backup("actor")
        companion.weight_version.fill_(2)
        companion.record_sample_consumptions([_identity(5, 0, 1)])

        assert ModelCompanionWeightVersionUtils.from_params(backuper.get("actor").items()) == 7
        backuper.restore("actor")

        assert companion.weight_version.item() == 7
        assert companion.snapshot_sample_consumptions(is_skipped=False) == {_identity(3, 0, 1): 1}
