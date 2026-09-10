"""Unit tests for LoRA-related helpers in miles.backends.megatron_utils.checkpoint.

Covers pure path-detection functions and the LoRA branch routing in
save_checkpoint_with_lora / load_checkpoint — the latter using mocks to avoid
GPU / distributed requirements.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.checkpoint import (
    _has_local_checkpoint,
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


class TestHasLocalCheckpoint:
    @staticmethod
    def _args(tmp_path, *, ckpt_step=None) -> Namespace:
        return Namespace(load=str(tmp_path), ckpt_step=ckpt_step, non_persistent_ckpt_type="local")

    def test_empty_manager_uses_the_disk_or_pretrained_fallback(self, tmp_path) -> None:
        """An allocated but empty local manager is not the selected checkpoint source."""
        manager = MagicMock()
        manager.find_latest.return_value = -1

        assert _has_local_checkpoint(self._args(tmp_path), {"local_checkpoint_manager": manager}) is False

    def test_older_manager_uses_the_newer_disk_checkpoint(self, tmp_path) -> None:
        """A local checkpoint older than the disk tracker is not selected."""
        manager = MagicMock()
        manager.find_latest.return_value = 7
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("8")

        assert _has_local_checkpoint(self._args(tmp_path), {"local_checkpoint_manager": manager}) is False

    def test_newer_manager_is_the_selected_checkpoint_source(self, tmp_path) -> None:
        """A local checkpoint at least as new as disk supplies the CPU witness."""
        manager = MagicMock()
        manager.find_latest.return_value = 8
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")

        assert _has_local_checkpoint(self._args(tmp_path), {"local_checkpoint_manager": manager}) is True

    def test_ckpt_step_takes_precedence_over_the_disk_tracker(self, tmp_path) -> None:
        """The requested persistent iteration determines whether local state wins."""
        manager = MagicMock()
        manager.find_latest.return_value = 8
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("9")

        assert _has_local_checkpoint(self._args(tmp_path, ckpt_step=7), {"local_checkpoint_manager": manager}) is True


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
