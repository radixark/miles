from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import model as model_module
from miles.backends.megatron_utils.ft import in_memory_checkpoint


class TestExplicitInMemoryConfig:
    def test_in_memory_save_runs_the_real_model_saver_with_the_supplied_trainer_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In-memory checkpoint creation threads the trainer config through the real save wrapper."""
        module = torch.nn.Linear(2, 2)
        module.lora_attached = False
        args = make_trainer_args(ci_test=False, use_distributed_optimizer=False, overlap_param_gather=False)
        expected = {"iteration": 17, "weight": module.weight.detach().clone()}
        observed: list[bool] = []
        monkeypatch.setattr(
            in_memory_checkpoint,
            "get_args",
            lambda: SimpleNamespace(non_persistent_ckpt_type="local", non_persistent_local_ckpt_algo="fully_parallel"),
        )

        def save_checkpoint(
            iteration: int, model: list, optimizer: object, opt_param_scheduler: object, **kwargs: object
        ) -> None:
            observed.append(kwargs["non_persistent_ckpt"])
            assert iteration == 17
            assert model == [module]
            kwargs["checkpointing_context"]["local_checkpoint_manager"].save(state_dict=expected, iteration=iteration)

        monkeypatch.setattr(model_module, "save_checkpoint", save_checkpoint)

        restored = in_memory_checkpoint.save_to_memory(
            args=args, iteration=17, model=[module], optimizer=None, opt_param_scheduler=None
        )

        assert observed == [True]
        assert restored["iteration"] == 17
        torch.testing.assert_close(restored["weight"], module.weight)
