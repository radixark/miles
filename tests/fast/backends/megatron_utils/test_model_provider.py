from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.model_provider import (
    LinearForLastLayer,
    _maybe_install_witness,
    _set_critic_output_layer,
)
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


class _LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_layer = torch.nn.Linear(4, 10)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.output_layer(hidden)
        return out[0] if isinstance(out, tuple) else out


class _VLMWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.language_model(hidden)


class TestCriticOutputLayer:
    config = SimpleNamespace(hidden_size=4, sequence_parallel=False)

    def test_value_head_goes_on_vlm_language_model(self) -> None:
        model = _VLMWrapper()

        _set_critic_output_layer(model, self.config)

        assert isinstance(model.language_model.output_layer, LinearForLastLayer)
        assert not hasattr(model, "output_layer")
        assert model(torch.zeros(3, 4)).shape == (3, 1)

    def test_value_head_replaces_plain_model_output_layer(self) -> None:
        model = _LanguageModel()

        _set_critic_output_layer(model, self.config)

        assert isinstance(model.output_layer, LinearForLastLayer)
        assert model(torch.zeros(3, 4)).shape == (3, 1)
