import json

import pytest
import safetensors.torch
import torch
from tools import convert_hf_to_fp8 as converter


@pytest.fixture
def convert_checkpoint(tmp_path, monkeypatch):
    safe_open = safetensors.safe_open
    monkeypatch.setattr(
        converter.safetensors,
        "safe_open",
        lambda path, framework, device: safe_open(path, framework=framework, device="cpu"),
    )
    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))

    def convert(weights, strategy, **kwargs):
        safetensors.torch.save_file(weights, source / "model.safetensors")
        converter.convert_fp8(
            str(source), str(target), strategy, block_size=[128, 128] if strategy == "block" else None, **kwargs
        )
        actual = safetensors.torch.load_file(target / "model.safetensors")
        config = json.loads((target / "config.json").read_text())["quantization_config"]
        ignored = config["ignore" if strategy == "channel" else "modules_to_not_convert"]
        index = json.loads((target / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == set(actual)
        return actual, ignored

    return convert


@pytest.mark.parametrize("strategy", ["block", "channel", "tensor"])
def test_default_scope_preserves_vision_and_non_matrix_parameters(convert_checkpoint, strategy):
    matrix = torch.arange(35, dtype=torch.bfloat16).reshape(5, 7) + 1
    quantized = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.shared_experts.down_proj.weight",
        "model.mtp.layers.0.self_attn.q_proj.weight",
    ]
    preserved = {
        "visual.blocks.0.attn.proj.weight": matrix.clone(),
        "model.visual.blocks.0.attn.qkv.weight": matrix.clone(),
        "vision_tower.blocks.0.attn.proj.weight": matrix.clone(),
        "model.mm_projector.weight": matrix.clone(),
        "model.layers.0.linear_attn.conv1d.weight": torch.ones(5, 1, 3, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.bias": torch.ones(5, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(5, dtype=torch.bfloat16),
        "model.embed_tokens.weight": matrix.clone(),
        "lm_head.weight": matrix.clone(),
    }
    weights = {name: matrix.clone() for name in quantized} | preserved
    actual, ignored = convert_checkpoint(weights, strategy)
    scale_suffix = "_scale_inv" if strategy == "block" else "_scale"
    assert set(actual) == set(weights) | {name + scale_suffix for name in quantized}
    assert set(ignored) == {name.removesuffix(".weight") for name in preserved if name.endswith(".weight")}
    for name in quantized:
        expected, scale = converter.quant_fp8(matrix, strategy, [128, 128] if strategy == "block" else None)
        assert actual[name].dtype == torch.float8_e4m3fn
        assert torch.equal(actual[name].view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(actual[name + scale_suffix], scale)
    for name, original in preserved.items():
        assert actual[name].dtype == original.dtype
        assert torch.equal(actual[name], original)


@pytest.mark.parametrize("strategy", ["block", "channel", "tensor"])
def test_extra_high_precision_layers_are_recorded_in_checkpoint(convert_checkpoint, strategy):
    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.q_proj.weight"]
    weights = {name: torch.ones(5, 7, dtype=torch.bfloat16) for name in names}
    actual, ignored = convert_checkpoint(weights, strategy, extra_high_precision_layers_hf=("model.layers.1.",))
    assert actual[names[0]].dtype == torch.float8_e4m3fn
    assert actual[names[1]].dtype == torch.bfloat16
    assert torch.equal(actual[names[1]], weights[names[1]])
    assert ignored == [names[1].removesuffix(".weight")]
    assert not any(key.startswith(names[1] + "_") for key in actual)
