from pathlib import Path

import pytest
from tests.fast.backends.megatron_utils.backend_scope_fixtures import _checkpoint_loader
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import checkpoint


class TestCheckpointMutationScope:
    @pytest.mark.parametrize("fails", [False, True])
    def test_native_checkpoint_sdk_updates_progress_only_within_the_load_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
    ) -> None:
        """Checkpoint progress updates survive loading while the SDK mutation lease always ends."""
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
        args = make_trainer_args(load=str(tmp_path), finetune=False, ckpt_step=None, lora_rank=0, spec=None)
        monkeypatch.setattr(
            checkpoint, "_load_checkpoint_megatron", _checkpoint_loader(backend=args.backend, fails=fails)
        )

        if fails:
            with pytest.raises(RuntimeError, match="checkpoint loading failed"):
                checkpoint.load_checkpoint([], None, None, None, False, args=args)
        else:
            result = checkpoint.load_checkpoint([], None, None, None, False, args=args)
            assert result.iteration == 7

        assert args.backend.consumed_train_samples == 64
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.consumed_train_samples = 65
