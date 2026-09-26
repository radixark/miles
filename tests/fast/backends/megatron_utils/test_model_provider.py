from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.model_provider import _maybe_install_witness
from miles.backends.training_utils.model_companion import ModelCompanion


class TestModelCompanionInstallation:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_companion_is_installed_independently_of_ownership_checking(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        """Every model includes the companion even when ownership recording is disabled."""
        group = SimpleNamespace(rank=0)
        monkeypatch.setattr(
            "miles.backends.training_utils.parallel.get_parallel_state",
            lambda: SimpleNamespace(pp=group, tp=group, cp=group, intra_dp=group),
        )
        args = Namespace(enable_sample_ownership_checker=enabled, enable_witness=False)
        model = torch.nn.Module()

        _maybe_install_witness(args=args, model=model, vp_stage=2)

        companions = [module for module in model.modules() if isinstance(module, ModelCompanion)]
        assert len(companions) == 1
        assert companions[0].chunk_index == 2
        assert companions[0].sample_consumptions.numel() == 0
