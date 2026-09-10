from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import _load_training_state, load_lora_adapter, save_lora_checkpoint
from miles.utils.audit_utils.witness.cpu import CpuWitness


class TestLoraCpuWitness:
    def test_native_adapter_restore_recovers_weight_history_without_an_optimizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabling optimizer restoration must not disable the adapter's weight witness restoration."""
        from megatron.bridge import AutoBridge
        from miles.backends.training_utils import parallel

        monkeypatch.setattr(
            parallel,
            "_parallel_state",
            Namespace(
                effective_dp=Namespace(rank=0), cp=Namespace(rank=0), tp=Namespace(rank=0), pp=Namespace(rank=0)
            ),
        )
        monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: None)
        model = torch.nn.Module()
        model.register_parameter("lora_weight", torch.nn.Parameter(torch.tensor([2.0])))
        witness = CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,))
        model.add_module("cpu_witness", witness)
        witness.commit(dict(rollout_id=0, group_indices=[7], slot=None))
        save_lora_checkpoint(
            model=[model], args=Namespace(hf_checkpoint="unused"), save_dir=str(tmp_path), iteration=0
        )
        with torch.no_grad():
            model.lora_weight.add_(10)
        witness.commit(dict(rollout_id=1, group_indices=[8], slot=None))

        loaded, iteration = load_lora_adapter(model=[model], adapter_path=str(tmp_path), optimizer=None)

        assert loaded and iteration == 0
        assert model.lora_weight.item() == 2.0
        assert [record["group_indices"] for record in witness.records] == [[7]]

    def test_new_checkpoint_missing_witness_is_rejected_even_without_optimizer(self, tmp_path: Path) -> None:
        """Missing witness metadata in a new checkpoint must not look like an empty training history."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (tmp_path / "cpu_witness_version.txt").write_text("1\n")
        torch.save(
            dict(iteration=1, optimizer=None, opt_param_scheduler=None), adapter_dir / "training_state_rank0.pt"
        )
        model = torch.nn.Module()
        model.add_module("cpu_witness", CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,)))

        with pytest.raises(AssertionError, match="CPU witness state is missing"):
            _load_training_state(adapter_dir=adapter_dir, optimizer=None, opt_param_scheduler=None, model=[model])
