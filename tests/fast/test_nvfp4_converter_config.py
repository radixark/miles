import json

import pytest
import safetensors.torch
import torch

from tools.convert_hf_to_nvfp4 import _get_expert_group_name, convert_nvfp4


@pytest.mark.parametrize(
    ("module_name", "expected_group"),
    [
        ("model.layers.0.mlp.experts.7.down_proj", "model.layers.0.mlp.experts"),
        ("model.layers.0.block_sparse_moe.experts.7.w1", "model.layers.0.block_sparse_moe.experts"),
        ("model.layers.0.moe.experts.7.w2", "model.layers.0.moe.experts"),
        ("model.layers.0.mlp.shared_experts.7.w3", "model.layers.0.mlp.shared_experts"),
    ],
)
def test_get_expert_group_name(module_name, expected_group):
    assert _get_expert_group_name(module_name) == expected_group


def test_bf16_moe_layer_uses_compact_ignore_prefixes(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')

    weights = {
        f"model.layers.0.mlp.experts.{expert_idx}.{projection}.weight": torch.ones((1, 16), dtype=torch.bfloat16)
        for expert_idx in range(128)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    safetensors.torch.save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})

    convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        num_layers_at_end_in_bf16=1,
    )

    expected_ignore = ["model.layers.0.", "model.layers.0.mlp.experts"]
    config = json.loads((save_dir / "config.json").read_text())
    assert config["quantization_config"]["ignore"] == expected_ignore

    hf_quant_config = json.loads((save_dir / "hf_quant_config.json").read_text())
    assert hf_quant_config["quantization"]["exclude_modules"] == expected_ignore

    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as f:
        assert all("weight_scale" not in key for key in f.keys())
        assert f.get_tensor("model.layers.0.mlp.experts.127.down_proj.weight").dtype == torch.bfloat16
