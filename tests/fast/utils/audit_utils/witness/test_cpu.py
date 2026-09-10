import pytest
import torch

from miles.utils.audit_utils.witness.cpu import (
    CpuWitness,
    TrainingSampleIdentity,
    hide_cpu_witness,
    preserve_cpu_witness,
    record_cpu_witness,
    snapshot_cpu_witness,
)


def _identity(source_sample_index: int, row_index: int, row_count: int) -> TrainingSampleIdentity:
    return TrainingSampleIdentity(
        source_sample_index=source_sample_index,
        row_index=row_index,
        row_count=row_count,
    )


class TestCpuWitness:
    def test_preserve_restores_state_after_other_model_load(self) -> None:
        """Auxiliary model loads cannot overwrite the actor witness."""
        model = torch.nn.Module()
        model.add_module("cpu_witness", CpuWitness())
        model.cpu_witness.record([_identity(7, 0, 1)])

        with preserve_cpu_witness([model]):
            model.cpu_witness.record([_identity(8, 0, 1)])

        assert model.cpu_witness.snapshot() == {_identity(7, 0, 1): 1}

    def test_hide_excludes_witness_from_legacy_checkpoint_load(self) -> None:
        """Legacy checkpoints can load while the new module is hidden."""
        model = torch.nn.Module()
        model.add_module("cpu_witness", CpuWitness())

        with hide_cpu_witness([model]):
            assert "cpu_witness" not in dict(model.named_children())

        assert "cpu_witness" in dict(model.named_children())

    def test_counts_every_occurrence(self) -> None:
        """Repeated sample identities remain visible as repeated consumption."""
        witness = CpuWitness()

        witness.record([_identity(7, 0, 1), _identity(8, 0, 1), _identity(7, 0, 1)])

        assert witness.snapshot() == {_identity(7, 0, 1): 2, _identity(8, 0, 1): 1}

    def test_extra_state_round_trip_replaces_counts(self) -> None:
        """Checkpoint restore replaces the current witness state."""
        source = CpuWitness()
        source.record([_identity(7, 0, 2), _identity(7, 1, 2), _identity(7, 0, 2)])
        target = CpuWitness()
        target.record([_identity(9, 0, 1)])

        target.set_extra_state(source.get_extra_state())

        assert target.snapshot() == {_identity(7, 0, 2): 2, _identity(7, 1, 2): 1}

    def test_model_chunks_receive_identical_updates(self) -> None:
        """Virtual pipeline chunks retain identical witness mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("cpu_witness", CpuWitness())

        record_cpu_witness(chunks, [_identity(7, 0, 2), _identity(7, 1, 2)])

        assert snapshot_cpu_witness(chunks) == {_identity(7, 0, 2): 1, _identity(7, 1, 2): 1}

    def test_snapshot_rejects_divergent_model_chunks(self) -> None:
        """A snapshot cannot silently choose between divergent chunk mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("cpu_witness", CpuWitness())
        chunks[0].cpu_witness.record([_identity(7, 0, 1)])
        chunks[1].cpu_witness.record([_identity(8, 0, 1)])

        with pytest.raises(AssertionError, match="chunks diverged"):
            snapshot_cpu_witness(chunks)
