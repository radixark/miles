"""Unit tests for miles.utils.hf_config."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _write_config_json(directory: str, config_dict: dict) -> None:
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)


class TestLoadHfConfig:
    def test_overrides_apply_to_returned_config(self, tmp_path):
        from miles.utils.hf_config import load_hf_config

        fake_config = SimpleNamespace(max_position_embeddings=4096, hidden_size=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_config):
            cfg = load_hf_config(
                str(tmp_path),
                overrides={"max_position_embeddings": 8192, "_attn_implementation": "flash"},
            )
        assert cfg.max_position_embeddings == 8192
        assert cfg.hidden_size == 128
        assert cfg._attn_implementation == "flash"

    def test_trust_remote_code_default_is_true(self, tmp_path):
        from miles.utils.hf_config import load_hf_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()) as mock_from_pretrained:
            load_hf_config(str(tmp_path))
        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["trust_remote_code"] is True

    def test_extra_kwargs_forwarded_to_autoconfig(self, tmp_path):
        from miles.utils.hf_config import load_hf_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()) as mock_from_pretrained:
            load_hf_config(str(tmp_path), revision="main", trust_remote_code=False)
        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["revision"] == "main"
        assert kwargs["trust_remote_code"] is False

    def test_unknown_model_type_raises(self, tmp_path):
        """Unrecognized model_type must fail loud, not be silently routed elsewhere."""
        from miles.utils.hf_config import load_hf_config

        _write_config_json(str(tmp_path), {"model_type": "totally_unknown_model"})
        with pytest.raises(ValueError):
            load_hf_config(str(tmp_path))

    def test_repeated_calls_are_idempotent(self, tmp_path):
        """Alias registration on every call must not raise on the second pass."""
        from miles.utils.hf_config import load_hf_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()):
            load_hf_config(str(tmp_path))
            load_hf_config(str(tmp_path))

    def test_native_support_raises_without_override(self, tmp_path):
        """If transformers ships native support for an alias, registration must fail loud."""
        from miles.utils import hf_config as hf_config_module

        alias = hf_config_module._HFConfigAlias(
            model_type="fake_native_type",
            base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
            base_class="DeepseekV3Config",
            compat_class_name="FakeNativeConfig",
        )
        with (
            patch.object(hf_config_module, "_CONFIG_ALIASES", (alias,)),
            patch.object(hf_config_module, "_REGISTERED_ALIASES", set()),
            patch.object(hf_config_module, "CONFIG_MAPPING_NAMES", {"fake_native_type": "FakeConfig"}),
            pytest.raises(RuntimeError, match="natively supports"),
        ):
            hf_config_module.load_hf_config(str(tmp_path))

    def test_native_support_overridden_when_flag_set(self, tmp_path):
        """override_hf_native=True must allow registration to win over native support."""
        from miles.utils import hf_config as hf_config_module

        alias = hf_config_module._HFConfigAlias(
            model_type="fake_native_type",
            base_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
            base_class="DeepseekV3Config",
            compat_class_name="FakeNativeConfig",
            override_hf_native=True,
        )
        with (
            patch.object(hf_config_module, "_CONFIG_ALIASES", (alias,)),
            patch.object(hf_config_module, "_REGISTERED_ALIASES", set()),
            patch.object(hf_config_module, "CONFIG_MAPPING_NAMES", {"fake_native_type": "FakeConfig"}),
            patch("transformers.AutoConfig.register") as mock_register,
            patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()),
        ):
            hf_config_module.load_hf_config(str(tmp_path))
        _, kwargs = mock_register.call_args
        assert kwargs["exist_ok"] is True


class TestDeepseekV32Alias:
    """Integration: alias registration makes AutoConfig recognize deepseek_v32."""

    def test_deepseek_v32_loads_via_alias(self, tmp_path):
        """A config.json with model_type=deepseek_v32 should load as a DeepseekV3Config subclass."""
        pytest.importorskip("transformers.models.deepseek_v3.configuration_deepseek_v3")
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        from miles.utils.hf_config import load_hf_config

        # Use DeepseekV3Config's __init__ defaults to produce a valid config dict,
        # then re-stamp the model_type as deepseek_v32.
        base_dict = DeepseekV3Config().to_dict()
        base_dict["model_type"] = "deepseek_v32"
        _write_config_json(str(tmp_path), base_dict)

        cfg = load_hf_config(str(tmp_path))
        assert cfg.model_type == "deepseek_v32"
        assert isinstance(cfg, DeepseekV3Config)

    @pytest.mark.skip(reason="FIXME: re-enable after deepseek_v32 AutoModel alias registration is fixed.")
    def test_deepseek_v32_resolves_via_auto_model_for_causal_lm(self, tmp_path):
        """The returned config must be resolvable by AutoModelForCausalLM.from_config.

        AutoConfig.register only updates CONFIG_MAPPING. AutoModelForCausalLM looks
        up the model class by exact config type in MODEL_FOR_CAUSAL_LM_MAPPING, so
        a synthesized DeepseekV32Config subclass that isn't separately registered
        there makes `AutoModelForCausalLM.from_config(cfg)` raise
        `ValueError: Unrecognized configuration class`. tools/convert_fsdp_to_hf.py
        depends on this path working.
        """
        pytest.importorskip("transformers.models.deepseek_v3.configuration_deepseek_v3")
        from transformers import AutoModelForCausalLM
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        from miles.utils.hf_config import load_hf_config

        base_dict = DeepseekV3Config().to_dict()
        base_dict["model_type"] = "deepseek_v32"
        _write_config_json(str(tmp_path), base_dict)

        cfg = load_hf_config(str(tmp_path))
        assert type(cfg) in AutoModelForCausalLM._model_mapping.keys(), (
            f"{type(cfg).__name__} not registered with AutoModelForCausalLM — "
            f"AutoModelForCausalLM.from_config(cfg) will raise ValueError."
        )
