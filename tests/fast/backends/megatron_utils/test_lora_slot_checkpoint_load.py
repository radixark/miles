"""load_slot validates every shard before it touches the live slot."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.lora import checkpoint as ckpt


def _write_weight_shard(tmp_path):
    with patch.object(ckpt, "get_parallel_state") as parallel_state:
        parallel_state.return_value = MagicMock(
            tp=MagicMock(rank=0), pp=MagicMock(rank=0), ep=MagicMock(rank=0, size=1)
        )
        torch.save({"layer.adapter.weight": torch.zeros(2)}, tmp_path / ckpt._weight_shard_name())
        return ckpt._weight_shard_name()


def test_a_missing_optim_shard_leaves_the_slot_untouched(tmp_path):
    weight_name = _write_weight_shard(tmp_path)
    optimizer = MagicMock()
    with (
        patch.object(ckpt, "_weight_shard_name", return_value=weight_name),
        patch.dict("sys.modules", {"megatron.bridge.peft.multi_lora_layers": (bridge := MagicMock())}),
    ):
        with pytest.raises(RuntimeError, match="FileNotFoundError"):
            ckpt.load_slot([MagicMock()], optimizer, slot=0, path=str(tmp_path), load_optimizer=True)
    bridge.load_adapter.assert_not_called()
    optimizer.reload_model_params.assert_not_called()


def test_a_layout_mismatch_is_rejected_before_apply(tmp_path):
    weight_name = _write_weight_shard(tmp_path)
    torch.save({"world_size": 8, "children": []}, tmp_path / ckpt._optim_shard_name())
    optimizer = MagicMock()
    with (
        patch.object(ckpt, "_weight_shard_name", return_value=weight_name),
        patch.dict("sys.modules", {"megatron.bridge.peft.multi_lora_layers": (bridge := MagicMock())}),
    ):
        with pytest.raises(RuntimeError, match="world_size"):
            ckpt.load_slot([MagicMock()], optimizer, slot=0, path=str(tmp_path), load_optimizer=True)
    bridge.load_adapter.assert_not_called()
    optimizer.reload_model_params.assert_not_called()
