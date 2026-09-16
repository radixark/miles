from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import checkpoint


class TestCheckpointTrainingProvenance:
    @pytest.mark.parametrize(
        ("source", "finetune", "ckpt_step", "local_iteration", "expected"),
        [
            ("0", False, None, None, True),
            ("release", False, None, None, True),
            ("0", True, None, None, False),
            ("iter_0000000", False, None, None, True),
            ("release", False, 0, None, True),
            ("release", False, None, -1, True),
            ("release", False, None, 0, True),
        ],
    )
    def test_native_load_uses_finetune_for_rollout_numbering(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source: str,
        finetune: bool,
        ckpt_step: int | None,
        local_iteration: int | None,
        expected: bool,
    ) -> None:
        """Native loading preserves the existing finetune-based rollout numbering."""
        load_path = tmp_path / (source if source.startswith("iter_") else "checkpoint")
        load_path.mkdir()
        if source.startswith("iter_"):
            (load_path / "metadata.json").write_text("{}")
        else:
            (load_path / "latest_checkpointed_iteration.txt").write_text(source)
        args = Namespace(load=str(load_path), finetune=finetune, ckpt_step=ckpt_step, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (0, 123))
        checkpointing_context = (
            None
            if local_iteration is None
            else {"local_checkpoint_manager": SimpleNamespace(find_latest=lambda: local_iteration)}
        )

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, expected, False)

    def test_loading_hf_weights_preserves_non_finetune_rollout_numbering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HF loading preserves the existing non-finetune rollout numbering."""
        (tmp_path / "config.json").write_text("{}")
        args = Namespace(load=str(tmp_path), finetune=False, ckpt_step=None, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: (0, 0))

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, True, False)

    @pytest.mark.parametrize(
        ("adapter_result", "expected"),
        [
            ((True, 0, True), True),
            ((True, 0, False), True),
            ((True, None, False), False),
            ((False, None, False), False),
        ],
    )
    def test_lora_restore_preserves_the_difference_between_zero_and_missing_training_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        adapter_result: tuple[bool, int | None, bool],
        expected: bool,
    ) -> None:
        """Adapter weights alone cannot be mistaken for a saved iteration-zero training state."""
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
        args = Namespace(
            load=str(tmp_path),
            finetune=True,
            ckpt_step=None,
            lora_rank=8,
            lora_adapter_path=str(tmp_path / "adapter"),
            no_load_optim=False,
        )
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (0, 123))
        monkeypatch.setattr(checkpoint, "load_lora_adapter", lambda *_args, **_kwargs: adapter_result)

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, expected, adapter_result[2])
