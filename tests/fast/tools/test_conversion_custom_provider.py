import sys
from types import SimpleNamespace
from typing import Annotated

import pytest
from tools.convert_hf_to_torch_dist import _ConversionConfig, get_args

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.function_registry import function_registry


class _ProviderConfig(BaseConfig):
    conversion_width: Annotated[int, Arg()] = 3


class TestConversionCustomProvider:
    @pytest.mark.parametrize("provider", [None, "conversion-test-provider"])
    def test_real_parser_separates_custom_model_inputs_from_backend_checkpoint_fields(
        self, monkeypatch: pytest.MonkeyPatch, provider: str | None
    ) -> None:
        """Conversion hooks and checkpoint fields retain their owners through typed serialization."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.delenv("MILES_BACKEND", raising=False)
        argv = [
            "convert",
            "--hf-checkpoint",
            "/input",
            "--save",
            "/output",
            "--num-layers",
            "2",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "2",
        ]
        if provider is not None:
            argv.extend(["--custom-model-provider-path", provider, "--conversion-width", "19"])
        monkeypatch.setattr(sys, "argv", argv)
        with function_registry.temporary("conversion-test-provider", SimpleNamespace(config_class=_ProviderConfig)):
            config = get_args()
            restored = _ConversionConfig.model_validate_json(config.model_dump_json())

        assert config.hf_checkpoint == restored.hf_checkpoint == "/input"
        assert config.backend.save == restored.backend.save == "/output"
        assert config.backend.num_layers == restored.backend.num_layers == 2
        assert config.trainer_id == "actor"
        assert config.multi_lora_n_adapters == 0
        assert "conversion_width" not in vars(config.backend)
        assert "save" not in _ConversionConfig.model_fields
        if provider is None:
            assert config.custom_model_provider_path is None
        else:
            assert isinstance(config.custom_model_provider_path, CustomFunctionConfig)
            assert config.custom_model_provider_path.path == provider
            assert config.custom_model_provider_path.config.conversion_width == 19
            assert restored.custom_model_provider_path.config.conversion_width == 19
