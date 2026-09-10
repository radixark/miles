import pytest
import torch

from miles.backends.training_utils.weight_companion import (
    WeightCompanion,
    TrainingSampleIdentity,
    hide_weight_companion,
    install_weight_companion,
    preserve_weight_companion,
    record_weight_companion,
    record_nonfinite_skip_weight_companion,
    snapshot_weight_companion,
    snapshot_nonfinite_skip_weight_companion,
)


def _identity(source_sample_index: int, row_index: int, row_count: int) -> TrainingSampleIdentity:
    return TrainingSampleIdentity(
        source_sample_index=source_sample_index,
        row_index=row_index,
        row_count=row_count,
    )


class TestWeightCompanion:
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

        install_weight_companion(model, chunk_index=1)

        assert model.cpu_witness.replica_id == (3, 4, 5)

    def test_preserve_restores_state_after_other_model_load(self) -> None:
        """Auxiliary model loads cannot overwrite the actor witness."""
        model = torch.nn.Module()
        model.add_module("cpu_witness", WeightCompanion())
        model.cpu_witness.record([_identity(7, 0, 1)])
        model.cpu_witness.record_skipped_nonfinite([_identity(8, 0, 1)])

        with preserve_weight_companion([model]):
            model.cpu_witness.record([_identity(9, 0, 1)])
            model.cpu_witness.record_skipped_nonfinite([_identity(10, 0, 1)])

        assert model.cpu_witness.snapshot() == {_identity(7, 0, 1): 1}
        assert model.cpu_witness.snapshot_skipped_nonfinite() == {_identity(8, 0, 1): 1}

    def test_hide_excludes_witness_from_legacy_checkpoint_load(self) -> None:
        """Legacy checkpoints can load while the new module is hidden."""
        model = torch.nn.Module()
        model.add_module("cpu_witness", WeightCompanion())

        with hide_weight_companion([model]):
            assert "cpu_witness" not in dict(model.named_children())

        assert "cpu_witness" in dict(model.named_children())

    def test_counts_every_occurrence(self) -> None:
        """Repeated sample identities remain visible as repeated consumption."""
        witness = WeightCompanion()

        witness.record([_identity(7, 0, 1), _identity(8, 0, 1), _identity(7, 0, 1)])

        assert witness.snapshot() == {_identity(7, 0, 1): 2, _identity(8, 0, 1): 1}

    def test_extra_state_round_trip_replaces_counts(self) -> None:
        """Checkpoint restore replaces the current witness state."""
        source = WeightCompanion()
        source.record([_identity(7, 0, 2), _identity(7, 1, 2), _identity(7, 0, 2)])
        source.record_skipped_nonfinite([_identity(8, 0, 1)])
        target = WeightCompanion()
        target.record([_identity(9, 0, 1)])
        target.record_skipped_nonfinite([_identity(10, 0, 1)])

        target.set_extra_state(source.get_extra_state())

        assert target.snapshot() == {_identity(7, 0, 2): 2, _identity(7, 1, 2): 1}
        assert target.snapshot_skipped_nonfinite() == {_identity(8, 0, 1): 1}

    def test_version_one_checkpoint_loads_with_no_nonfinite_skips(self) -> None:
        """Older witness checkpoints restore trained counts and an empty skip counter."""
        witness = WeightCompanion()
        witness.record_skipped_nonfinite([_identity(8, 0, 1)])

        witness.set_extra_state({"version": 1, "sample_counts": {_identity(7, 0, 1): 1}})

        assert witness.snapshot() == {_identity(7, 0, 1): 1}
        assert witness.snapshot_skipped_nonfinite() == {}

    def test_version_two_checkpoint_requires_the_nonfinite_skip_counter(self) -> None:
        """A truncated current checkpoint cannot silently erase nonfinite outcomes."""
        witness = WeightCompanion()

        with pytest.raises(KeyError, match="skipped_nonfinite_sample_counts"):
            witness.set_extra_state({"version": 2, "sample_counts": {}})

    def test_model_chunks_receive_identical_updates(self) -> None:
        """Virtual pipeline chunks retain identical witness mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("cpu_witness", WeightCompanion())

        record_weight_companion(chunks, [_identity(7, 0, 2), _identity(7, 1, 2)])
        record_nonfinite_skip_weight_companion(chunks, [_identity(8, 0, 1)])

        assert snapshot_weight_companion(chunks) == {_identity(7, 0, 2): 1, _identity(7, 1, 2): 1}
        assert snapshot_nonfinite_skip_weight_companion(chunks) == {_identity(8, 0, 1): 1}

    def test_snapshot_rejects_divergent_model_chunks(self) -> None:
        """A snapshot cannot silently choose between divergent chunk mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("cpu_witness", WeightCompanion())
        chunks[0].cpu_witness.record([_identity(7, 0, 1)])
        chunks[1].cpu_witness.record([_identity(8, 0, 1)])

        with pytest.raises(AssertionError, match="chunks diverged"):
            snapshot_weight_companion(chunks)
