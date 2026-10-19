import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import Qwen3MoeConfig

from miles.utils.hf_utils.weight_mapping import HfWeightMapping, get_param_name_remap


@pytest.fixture(scope="module")
def hf_mapping():
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
    )
    return HfWeightMapping.from_config(config)


_PREFIX = "model.layers.0.mlp.experts"
_TARGETS = [f"{_PREFIX}.gate_up_proj", f"{_PREFIX}.down_proj"]


def _unpacked_names():
    return {
        f"{_PREFIX}.{expert}.{projection}_proj.weight" for expert in range(2) for projection in ("gate", "up", "down")
    }


def test_checkpoint_formats_resolve_to_the_same_hf_parameters(hf_mapping):
    unpacked = _unpacked_names()
    assert {hf_mapping.model_parameter(name) for name in unpacked} == set(_TARGETS)
    assert {hf_mapping.model_parameter(name) for name in _TARGETS} == set(_TARGETS)


@pytest.mark.parametrize("name", ["layers.0.attn.wq_a.weight", "mtp.0.emb.tok_emb.weight", "mtp.0.enorm.weight"])
def test_native_dsv4_remap_passes_nextn_context(tmp_path, monkeypatch, name):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"], "num_hidden_layers": 43}))
    remap = Mock(return_value="mapped")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.models.deepseek_v4",
        SimpleNamespace(DeepseekV4ForCausalLM=SimpleNamespace(remap_weight_name_to_dpsk_hf_format=remap)),
    )

    assert get_param_name_remap(str(config), {"embed.weight": "model.safetensors"})(name) == "mapped"
    remap.assert_called_once_with(name, is_nextn=True, num_hidden_layers=43)


def test_canonical_checkpoint_names_are_preserved(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}))
    name = "model.embed_tokens.weight"
    assert get_param_name_remap(str(config), {name: "model.safetensors"})(name) == name
