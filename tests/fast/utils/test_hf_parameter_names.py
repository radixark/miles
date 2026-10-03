import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.utils.hf_parameter_names import get_param_name_remap


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
