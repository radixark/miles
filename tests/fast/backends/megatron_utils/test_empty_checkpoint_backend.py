from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.megatron_utils.sdk_config_fixtures import _ModelSdk, model_sdk
from tests.fast.fixtures.args_fixtures import make_trainer_config

__all__ = ["model_sdk"]


class TestEmptyCheckpointBackendPolicy:
    @pytest.mark.parametrize("finetune,expected_start", [(True, 0), (False, 1)])
    def test_empty_checkpoint_preserves_backend_finetune_rollout_policy(
        self,
        model_sdk: _ModelSdk,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        finetune: bool,
        expected_start: int,
    ) -> None:
        """An empty checkpoint directory retains the existing finetune-dependent rollout numbering policy."""
        args = make_trainer_config(
            load=str(tmp_path),
            finetune=finetune,
            lora_rank=0,
            multi_lora=False,
            ci_test=False,
            use_checkpoint_opt_param_scheduler=True,
            global_batch_size=12,
        )
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
        scheduler = Mock()

        result = model_sdk.module.load_model_state(
            args,
            model=[model_sdk.model],
            optimizer=None,
            opt_param_scheduler=scheduler,
            role="actor",
            checkpointing_context=None,
            resume_from_ckpt=True,
        )

        assert result.loaded_rollout_id == 0
        assert result.start_rollout_id == expected_start
        scheduler.step.assert_called_once_with(increment=0)
