from pathlib import Path

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import checkpoint
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


class TestCheckpointDeserialization:
    @pytest.mark.parametrize("fail_after_load", [False, True])
    def test_backend_namespace_is_allowlisted_only_during_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after_load: bool
    ) -> None:
        """Weights-only checkpoint loading accepts typed backend arguments without leaking the allowlist."""
        path = tmp_path / "args.pt"
        torch.save(MegatronArgsNamespace(micro_batch_size=3), path)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
        args = make_trainer_args(load=str(tmp_path), finetune=False, ckpt_step=None, lora_rank=0, spec=None)
        previous = set(torch.serialization.get_safe_globals())
        loaded: list[MegatronArgsNamespace] = []

        def load(**kwargs: object) -> tuple[int, int]:
            value = torch.load(path, weights_only=True)
            assert isinstance(value, MegatronArgsNamespace)
            assert value.micro_batch_size == 3
            loaded.append(value)
            if fail_after_load:
                raise RuntimeError("loader failed after deserialization")
            return 7, 0

        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", load)

        if fail_after_load:
            with pytest.raises(RuntimeError, match="loader failed after deserialization"):
                checkpoint.load_checkpoint([], None, None, None, False, args=args)
        else:
            result = checkpoint.load_checkpoint([], None, None, None, False, args=args)
            assert result == (7, True)

        assert len(loaded) == 1
        assert set(torch.serialization.get_safe_globals()) == previous
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.load = "leaked"
