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

    def test_namespace_fallback_when_autoconfig_raises(self, tmp_path):
        """Unknown model_type should fall back to a namespace built from config.json."""
        import torch

        from miles.utils.hf_config import load_hf_config

        _write_config_json(
            str(tmp_path),
            {"model_type": "totally_unknown", "hidden_size": 1024, "torch_dtype": "bfloat16"},
        )

        cfg = load_hf_config(str(tmp_path))
        assert cfg.model_type == "totally_unknown"
        assert cfg.hidden_size == 1024
        # bfloat16 string in config.json should have been mapped to a torch dtype
        assert cfg.torch_dtype is torch.bfloat16

    def test_namespace_fallback_with_overrides(self, tmp_path):
        from miles.utils.hf_config import load_hf_config

        _write_config_json(str(tmp_path), {"model_type": "weird", "hidden_size": 1024})
        cfg = load_hf_config(str(tmp_path), overrides={"hidden_size": 2048})
        assert cfg.hidden_size == 2048

    def test_namespace_fallback_handles_text_config(self, tmp_path):
        """Multimodal configs with nested text_config should be wrapped too."""
        from miles.utils.hf_config import load_hf_config

        _write_config_json(
            str(tmp_path),
            {
                "model_type": "weird_multimodal",
                "text_config": {"hidden_size": 4096, "torch_dtype": "float16"},
            },
        )
        cfg = load_hf_config(str(tmp_path))
        assert cfg.text_config.hidden_size == 4096

        import torch

        assert cfg.text_config.torch_dtype is torch.float16

    def test_repeated_calls_are_idempotent(self, tmp_path):
        """Alias registration on every call must not raise on the second pass."""
        from miles.utils.hf_config import load_hf_config

        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace()):
            load_hf_config(str(tmp_path))
            load_hf_config(str(tmp_path))


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
