from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.provider_signature_fixtures import provider_parallel_state
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.megatron_utils.model_provider import _maybe_install_witness, get_model_provider_func
from miles.backends.training_utils.model_companion import ModelCompanion
from miles.utils.args.trainer_utils import compute_trainer_config

_ = provider_parallel_state


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


class TestCustomProviderSignatures:
    @pytest.mark.parametrize(
        "provider_name,expected_rate,expected_stage",
        [
            ("_legacy_provider", None, None),
            ("_provider_with_args", 0.003, None),
            ("_provider_with_stage", None, 3),
            ("_provider_with_args_and_stage", 0.003, 3),
        ],
    )
    def test_optional_configuration_and_pipeline_stage_are_forwarded_only_when_declared(
        self,
        provider_parallel_state: None,
        provider_name: str,
        expected_rate: float | None,
        expected_stage: int | None,
    ) -> None:
        """Custom providers receive their declared configuration and pipeline inputs."""
        path = f"tests.fast.backends.megatron_utils.provider_signature_fixtures.{provider_name}"
        config = parse_megatron_test_config("--custom-model-provider-path", path, "--lr", "0.003")
        trainer = compute_trainer_config(all_config=config, trainer=config.raw_megatron.trainers[0])
        provider = get_model_provider_func(args=trainer)

        model = provider(pre_process=False, post_process=True, vp_stage=3)

        assert model.pre_process is False
        assert model.post_process is True
        assert model.learning_rate == expected_rate
        assert model.vp_stage == expected_stage
