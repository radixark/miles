import json
from pathlib import Path

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils.model import save


class TestExplicitSaveConfig:
    @pytest.mark.parametrize("omit_optimizer", [False, True])
    def test_model_save_uses_the_supplied_config_through_the_lora_checkpoint_writer(
        self, tmp_path: Path, omit_optimizer: bool
    ) -> None:
        """The supplied trainer config determines the adapter destination and optimizer persistence."""
        model = torch.nn.Linear(2, 2)
        model.lora_attached = True
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        args = make_trainer_args(
            save=str(tmp_path / "requested"),
            lora_rank=4,
            lora_alpha=9,
            lora_adapter_targets=("linear_qkv",),
            megatron_to_hf_mode="raw",
            no_save_optim=omit_optimizer,
            use_distributed_optimizer=False,
            overlap_param_gather=False,
            ci_test=False,
        )

        save(args=args, iteration=13, model=[model], optimizer=optimizer, opt_param_scheduler=None)

        adapter_dir = tmp_path / "requested" / "iter_0000013" / "adapter"
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert config["r"] == 4
        assert config["lora_alpha"] == 9
        state = torch.load(adapter_dir / "training_state_rank0.pt", weights_only=False)
        assert state["iteration"] == 13
        assert (state["optimizer"] is None) is omit_optimizer
        if not omit_optimizer:
            assert state["optimizer"]["param_groups"][0]["lr"] == 0.25
        assert args.backend.save == str(tmp_path / "requested")
