"""Unit tests for LoRA-related helpers in miles.backends.megatron_utils.checkpoint.

Covers pure path-detection functions and the LoRA branch routing in
save_checkpoint_with_lora / load_checkpoint — the latter using mocks to avoid
GPU / distributed requirements.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.checkpoint import (
    _exclude_adapter_params_from_sharded_state_dict,
    _is_megatron_checkpoint,
    save_checkpoint_with_lora,
)

# ---------------------------------------------------------------------------
# _is_megatron_checkpoint
# ---------------------------------------------------------------------------


class TestIsMegatronCheckpoint:
    def test_has_latest_file(self, tmp_path):
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
        assert _is_megatron_checkpoint(tmp_path) is True

    def test_iter_dir_name(self, tmp_path):
        iter_dir = tmp_path / "iter_0000100"
        iter_dir.mkdir()
        assert _is_megatron_checkpoint(iter_dir) is True

    def test_regular_dir(self, tmp_path):
        assert _is_megatron_checkpoint(tmp_path) is False

    def test_hf_checkpoint_dir(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")
        assert _is_megatron_checkpoint(tmp_path) is False

    @pytest.mark.parametrize(
        "name",
        [
            "iter_0000001",
            "iter_0000000",
            "iter_9999999",
        ],
    )
    def test_valid_iter_patterns(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is True

    @pytest.mark.parametrize(
        "name",
        [
            "iter_123",  # too short
            "iter_00000001",  # too long
            "iteration_0000001",
            "checkpoint",
        ],
    )
    def test_invalid_iter_patterns(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is False


# ---------------------------------------------------------------------------
# save_checkpoint_with_lora — branch routing
# ---------------------------------------------------------------------------


class TestSaveCheckpointWithLoRA:
    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_lora_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=True)
    def test_lora_model_saves_adapter(self, mock_is_lora, mock_save_lora, mock_get_args, tmp_path):
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock())

        mock_save_lora.assert_called_once()
        call_args = mock_save_lora.call_args
        assert "adapter" in call_args[1].get("save_dir", call_args[0][2] if len(call_args[0]) > 2 else "")

    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=False)
    def test_non_lora_model_saves_regular(self, mock_is_lora, mock_save_ckpt, mock_get_args, tmp_path):
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()


class TestExcludeAdapterParamsFromShardedStateDict:
    def test_filters_adapters_and_restores_method(self):
        wrapped_model = MagicMock()
        model = MagicMock()
        original = MagicMock(
            return_value={
                "decoder.layers.0.mlp.linear_fc1.weight": object(),
                "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight": object(),
                "decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight": object(),
            }
        )
        model.sharded_state_dict = original

        with patch(
            "miles.backends.megatron_utils.checkpoint.unwrap_model", return_value=[model]
        ), _exclude_adapter_params_from_sharded_state_dict([wrapped_model]):
            assert list(model.sharded_state_dict(metadata={})) == ["decoder.layers.0.mlp.linear_fc1.weight"]

        assert model.sharded_state_dict is original

    def test_restores_method_after_error(self):
        model = MagicMock()
        original = MagicMock(return_value={})
        model.sharded_state_dict = original

        with pytest.raises(RuntimeError), _exclude_adapter_params_from_sharded_state_dict([model]):
            raise RuntimeError("load failed")

        assert model.sharded_state_dict is original
