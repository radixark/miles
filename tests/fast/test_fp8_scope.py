"""Exercise FP8 checkpoint selection and exclusion metadata on CPU."""

import json

import pytest
import safetensors.torch
import torch
from tools import convert_hf_to_fp8 as converter


@pytest.fixture
def cpu_safetensors(monkeypatch):
    safe_open = safetensors.safe_open
    monkeypatch.setattr(
        converter.safetensors,
        "safe_open",
        lambda path, framework, device: safe_open(path, framework=framework, device="cpu"),
    )


def _convert(tmp_path, weights, strategy, **kwargs):
    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    safetensors.torch.save_file(weights, source / "model.safetensors")
    converter.convert_fp8(
        str(source), str(target), strategy, block_size=[128, 128] if strategy == "block" else None, **kwargs
    )
    actual = safetensors.torch.load_file(target / "model.safetensors")
    config = json.loads((target / "config.json").read_text())["quantization_config"]
    ignored = config.get("ignore" if strategy == "channel" else "modules_to_not_convert", [])
    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(actual)
    return actual, ignored


def _assert_unchanged(actual, name, original, ignored):
    assert actual[name].dtype == original.dtype
    assert torch.equal(actual[name], original)
    if not name.endswith((".bias", "_bias")):
        assert name.removesuffix(".weight") in ignored
    if name.endswith(".weight"):
        assert not any(key.startswith(name + "_") for key in actual)


@pytest.mark.parametrize("strategy", ["block", "channel", "tensor"])
def test_extra_high_precision_layers_preserve_weights_and_record_exclusions(tmp_path, cpu_safetensors, strategy):
    matrix = torch.arange(35, dtype=torch.bfloat16).reshape(5, 7) + 1
    quantized = "model.layers.0.mlp.experts.0.gate_proj.weight"
    preserved = {
        "visual.blocks.0.attn.proj.weight": matrix.clone(),
        "model.visual.blocks.0.attn.qkv.weight": matrix.clone(),
        "model.layers.0.linear_attn.conv1d.weight": torch.ones(5, 1, 3, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.bias": torch.ones(5, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(5, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.ones(2, 10, 7, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.ones(2, 7, 5, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.ones(2, 10, 7, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.experts.down_proj": torch.ones(2, 7, 5, dtype=torch.bfloat16),
        "model.language_model.layers.0.hc_attn_fn": matrix.clone(),
        "model.language_model.layers.0.linear_attn.A_log": torch.ones(5, dtype=torch.float32),
        "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(5, dtype=torch.float32),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.ones(5, dtype=torch.float32),
        "layers.0.attn.attn_sink": torch.ones(5, dtype=torch.float32),
        "model.layers.0.self_attn.rotary_emb.inv_freq": torch.ones(5, dtype=torch.float32),
    }
    weights = {quantized: matrix} | preserved
    actual, ignored = _convert(
        tmp_path, weights, strategy, extra_high_precision_layers_hf=("visual.", "linear_attn.conv1d")
    )
    scale_suffix = "_scale_inv" if strategy == "block" else "_scale"
    assert set(actual) == set(weights) | {quantized + scale_suffix}
    assert actual[quantized].dtype == torch.float8_e4m3fn
    assert set(ignored) == {
        name.removesuffix(".weight") for name in preserved if not name.endswith((".bias", "_bias"))
    }
    for name, original in preserved.items():
        _assert_unchanged(actual, name, original, ignored)


@pytest.mark.parametrize("strategy", ["block", "channel", "tensor"])
def test_no_extra_exclusions_preserves_existing_conversion(tmp_path, cpu_safetensors, strategy):
    matrix = torch.arange(35, dtype=torch.bfloat16).reshape(5, 7) + 1
    weights = {
        name: matrix.clone()
        for name in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.shared_experts.down_proj.weight",
            "model.mtp.layers.0.self_attn.q_proj.weight",
            "visual.blocks.0.attn.proj.weight",
        )
    }
    if strategy != "block":
        weights["model.layers.1.mlp.experts.gate_up_proj.weight"] = matrix.unsqueeze(0).repeat(2, 1, 1)
    actual, ignored = _convert(tmp_path, weights, strategy)
    assert ignored == []
    scale_suffix = "_scale_inv" if strategy == "block" else "_scale"
    for name, original in weights.items():
        expected, scale = converter.quant_fp8(original, strategy, [128, 128] if strategy == "block" else None)
        assert actual[name].dtype == torch.float8_e4m3fn
        assert torch.equal(actual[name].view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(actual[name + scale_suffix], scale)


def test_selected_unsupported_block_weight_raises(tmp_path, cpu_safetensors):
    weights = {"model.layers.0.linear_attn.conv1d.weight": torch.ones(5, 1, 3, dtype=torch.bfloat16)}
    with pytest.raises(ValueError, match="Block FP8 conversion requires a 2-D weight"):
        _convert(tmp_path, weights, "block")
