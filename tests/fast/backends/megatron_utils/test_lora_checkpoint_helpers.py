"""Unit tests for LoRA-related helpers in miles.backends.megatron_utils.checkpoint.

Covers pure path-detection functions and the LoRA branch routing in
save_checkpoint_with_lora / load_checkpoint — the latter using mocks to avoid
GPU / distributed requirements.
"""

from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.checkpoint import (
    _hide_bridge_lora_adapters_from_dist_checkpoint,
    _is_megatron_checkpoint,
    load_checkpoint,
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


class TestLoadBaseCheckpointWithLoRA:
    @staticmethod
    def _adapter_module():
        module = ModuleType("megatron.bridge.peft.adapter_wrapper")

        class AdapterWrapper:
            def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
                return {"adapter": prefix}

        module.AdapterWrapper = AdapterWrapper
        return module, AdapterWrapper

    def test_adapter_wrapper_exposes_base_state_and_restores_method(self):
        module, adapter_wrapper = self._adapter_module()
        original = adapter_wrapper.sharded_state_dict
        wrapper = adapter_wrapper()
        wrapper.to_wrap = MagicMock()
        wrapper.to_wrap.sharded_state_dict.return_value = {"base": "state"}

        with patch.dict("sys.modules", {module.__name__: module}):
            with _hide_bridge_lora_adapters_from_dist_checkpoint():
                result = wrapper.sharded_state_dict("model.", ((0, 1, 2),), {"key": "value"})

        assert result == {"base": "state"}
        wrapper.to_wrap.sharded_state_dict.assert_called_once_with("model.", ((0, 1, 2),), {"key": "value"})
        assert adapter_wrapper.sharded_state_dict is original

    def test_adapter_wrapper_method_is_restored_after_error(self):
        module, adapter_wrapper = self._adapter_module()
        original = adapter_wrapper.sharded_state_dict

        with patch.dict("sys.modules", {module.__name__: module}):
            with pytest.raises(RuntimeError, match="checkpoint load failed"):
                with _hide_bridge_lora_adapters_from_dist_checkpoint():
                    raise RuntimeError("checkpoint load failed")

        assert adapter_wrapper.sharded_state_dict is original

    @pytest.mark.parametrize("is_lora", [True, False])
    def test_megatron_load_hides_adapters_only_for_lora(self, tmp_path, is_lora):
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
        args = Namespace(load=str(tmp_path))
        load_context = MagicMock()

        with (
            patch("miles.backends.megatron_utils.checkpoint.get_args", return_value=args),
            patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=is_lora),
            patch("miles.backends.megatron_utils.checkpoint.is_lora_enabled", return_value=False),
            patch(
                "miles.backends.megatron_utils.checkpoint._hide_bridge_lora_adapters_from_dist_checkpoint",
                return_value=load_context,
            ) as hide_adapters,
            patch(
                "miles.backends.megatron_utils.checkpoint._load_checkpoint_megatron", return_value=(1, 0)
            ) as load_megatron,
        ):
            result = load_checkpoint(None, None, None, None, False)

        assert result == (1, 0)
        assert hide_adapters.call_count == int(is_lora)
        load_megatron.assert_called_once()
