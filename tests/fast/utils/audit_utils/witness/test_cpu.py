from pathlib import Path

import pytest
import torch

from miles.utils.audit_utils.witness.cpu import (
    CpuWitness,
    hide_cpu_witness,
    preserve_cpu_witness,
    restore_adapter_cpu_witness,
    snapshot_adapter_cpu_witness,
)


class TestCpuWitness:
    def test_serialized_weights_and_history_roll_back_together(self, tmp_path: Path) -> None:
        """Loading a model checkpoint restores the weight and its training evidence together."""
        model = torch.nn.Linear(1, 1)
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        model.add_module("cpu_witness", witness)
        witness.commit(dict(rollout_id=0, group_indices=[17], slot=None))
        path = tmp_path / "model.pt"
        original_weight = model.weight.detach().clone()
        torch.save(model.state_dict(), path)
        witness.commit(dict(rollout_id=1, group_indices=[23], slot=None))
        with torch.no_grad():
            model.weight.add_(1)

        model.load_state_dict(torch.load(path, weights_only=True))

        assert torch.equal(model.weight, original_weight)
        assert [record["group_indices"] for record in witness.records] == [[17]]
        assert list(witness.parameters()) == []
        assert list(witness.buffers()) == []

    def test_async_snapshot_does_not_alias_later_training(self) -> None:
        """A queued checkpoint cannot observe records or pending gradients added later."""
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        witness.commit(dict(rollout_id=0, group_indices=[17], slot=2), slot=2)
        snapshot = witness.get_extra_state()

        witness.step_slots([2])
        witness.records[0]["group_indices"].append(23)

        assert snapshot["records"] == []
        assert snapshot["pending"][2][0]["group_indices"] == [17]

    def test_only_stepped_adapter_gradients_become_weight_evidence(self) -> None:
        """Gradient accumulation owns groups without falsely claiming an optimizer update."""
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        witness.commit(dict(rollout_id=0, group_indices=[1], slot=2), slot=2)
        witness.commit(dict(rollout_id=1, group_indices=[2], slot=3), slot=3)
        witness.step_slots([2])

        assert [record["group_indices"] for record in witness.records] == [[1]]
        assert [record["group_indices"] for record in witness.pending[3]] == [[2]]

    def test_auxiliary_load_cannot_replace_actor_history_even_on_error(self) -> None:
        """An auxiliary weight load restores the actor witness when loading raises."""
        model = torch.nn.Module()
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        model.add_module("cpu_witness", witness)
        witness.commit(dict(rollout_id=0, group_indices=[1], slot=None))
        with pytest.raises(RuntimeError, match="load failed"):
            with preserve_cpu_witness([model]):
                witness.records.clear()
                raise RuntimeError("load failed")

        assert witness.records[0]["group_indices"] == [1]

    def test_legacy_load_can_hide_only_the_witness(self) -> None:
        """Legacy checkpoints load normally while the new metadata child is hidden."""
        model = torch.nn.Linear(1, 1)
        legacy = model.state_dict()
        model.add_module("cpu_witness", CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,)))
        with hide_cpu_witness([model]):
            model.load_state_dict(legacy)
        assert "cpu_witness._extra_state" in model.state_dict()
        with pytest.raises(RuntimeError, match="cpu_witness._extra_state"):
            model.load_state_dict(legacy)

    def test_pp_and_virtual_chunks_have_distinct_checkpoint_keys(self) -> None:
        """PP stages and VPP chunks cannot overwrite each other's metadata objects."""
        shards = [
            CpuWitness(pipeline_rank=pp, chunk_index=chunk, replica_id=(0, 0, 0)).sharded_state_dict(
                prefix="cpu_witness."
            )
            for pp in range(2)
            for chunk in range(2)
        ]
        objects = [shard["cpu_witness._extra_state"] for shard in shards]
        assert len({obj.key for obj in objects}) == 4
        assert all(obj.replica_id == (0, 0, 0) for obj in objects)

    def test_adapter_checkpoint_rejects_unsaved_gradients_and_remaps_its_slot(self) -> None:
        """Adapter restore retains mapping identity when an adapter moves into another slot."""
        model = torch.nn.Module()
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        model.add_module("cpu_witness", witness)
        witness.commit(dict(rollout_id=0, group_indices=[1], slot=2), slot=2)
        with pytest.raises(AssertionError, match="uncommitted gradients"):
            snapshot_adapter_cpu_witness(model=[model], slot=2)
        witness.step_slots([2])
        states = snapshot_adapter_cpu_witness(model=[model], slot=2)
        restore_adapter_cpu_witness(model=[model], states=states, slot=3)

        restored = snapshot_adapter_cpu_witness(model=[model], slot=3)
        assert restored[0]["records"][0]["slot"] == 2
        assert restored[0]["records"][0]["adapter_slot"] == 3
