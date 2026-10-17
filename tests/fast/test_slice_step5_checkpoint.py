"""Step5 checkpoint slicing on CPU: tensor renaming, dtype preservation and config rewriting."""

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from tools import slice_step5_checkpoint as slicer

# A 6-layer decoder plus 1 MTP layer: dense 0-2, MoE 3-5, full attention on 3 and the MTP layer is sliding.
_NUM_LAYERS = 6
_LAYER_TYPES = ["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 2 + ["sliding_attention"]


def _config() -> dict:
    return {
        "architectures": ["Step5ForConditionalGeneration"],
        "model_type": "step5_vl",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "model.layers.5.moe.experts.w13_weight",
                "model.layers.6.mtp_block.mlp.down_proj",
                "vision_model.transformer.resblocks.0.attn.qkv_proj",
                "lm_head",
            ],
        },
        "text_config": {
            "model_type": "step5",
            "num_hidden_layers": _NUM_LAYERS,
            "num_nextn_predict_layers": 1,
            "layer_types": _LAYER_TYPES,
            "partial_rotary_factors": [1.0, 1.0, 1.0, 1 / 3, 1.0, 1.0, 1.0],
            "rope_theta": [1e4, 1e4, 1e4, 1e7, 1e4, 1e4, 1e4],
            "swiglu_limits": [7.0 + i for i in range(_NUM_LAYERS + 1)],
            "swiglu_limits_shared": [7.0] * (_NUM_LAYERS + 1),
            "moe_layer_list": [3, 4, 5],
            "moe_layers_enum": "3,4,5",
        },
    }


def _tensors() -> dict[str, torch.Tensor]:
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "lm_head.weight": torch.randn(8, 4),
        "model.norm.weight": torch.randn(4),
        "vision_model.conv1.weight": torch.randn(2, 3),
        "model.layers.3.moe.gate_proj.weight": torch.randn(2, 4, 4).to(torch.float8_e4m3fn),
        "model.layers.3.moe.gate_proj.weight_scale_inv": torch.rand(2, 1, 1),
        "model.layers.5.moe.gate_proj.weight": torch.randn(2, 4, 4).to(torch.bfloat16),
        "model.layers.6.eh_proj.weight": torch.randn(4, 8),
    }
    for layer in range(_NUM_LAYERS + 1):
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.full((4,), float(layer))
    return tensors


def _write_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    directory.mkdir()
    tensors = _tensors()
    names = sorted(tensors)
    shards = {"a.safetensors": names[: len(names) // 2], "b.safetensors": names[len(names) // 2 :]}
    weight_map = {}
    for file_name, shard_names in shards.items():
        safetensors.torch.save_file({n: tensors[n] for n in shard_names}, str(directory / file_name))
        weight_map.update({n: file_name for n in shard_names})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps(_config()))
    (directory / "tokenizer.json").write_text("{}")
    return tensors


def _read_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for file_name in set(weight_map.values()):
        out.update(safetensors.torch.load_file(str(directory / file_name)))
    assert set(out) == set(weight_map)
    return out


def test_slice_renames_layers_and_keeps_dtypes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    original = _write_checkpoint(src)

    slicer.slice_checkpoint(src=src, dst=dst, layers=[0, 3, 5])
    sliced = _read_checkpoint(dst)

    expected = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "model.norm.weight",
        "vision_model.conv1.weight": "vision_model.conv1.weight",
        "model.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        "model.layers.1.input_layernorm.weight": "model.layers.3.input_layernorm.weight",
        "model.layers.2.input_layernorm.weight": "model.layers.5.input_layernorm.weight",
        "model.layers.1.moe.gate_proj.weight": "model.layers.3.moe.gate_proj.weight",
        "model.layers.1.moe.gate_proj.weight_scale_inv": "model.layers.3.moe.gate_proj.weight_scale_inv",
        "model.layers.2.moe.gate_proj.weight": "model.layers.5.moe.gate_proj.weight",
    }
    assert set(sliced) == set(expected)
    for new_name, old_name in expected.items():
        assert sliced[new_name].dtype == original[old_name].dtype
        assert torch.equal(sliced[new_name].view(torch.uint8), original[old_name].view(torch.uint8))
    assert (dst / "tokenizer.json").exists()

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in sliced.values())


def test_slice_rewrites_config(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_checkpoint(src)

    slicer.slice_checkpoint(src=src, dst=dst, layers=[0, 3, 5])
    config = json.loads((dst / "config.json").read_text())
    text = config["text_config"]

    assert text["num_hidden_layers"] == 3
    assert text["num_nextn_predict_layers"] == 0
    assert text["layer_types"] == ["sliding_attention", "full_attention", "sliding_attention"]
    assert text["partial_rotary_factors"] == [1.0, 1 / 3, 1.0]
    assert text["rope_theta"] == [1e4, 1e7, 1e4]
    assert text["swiglu_limits"] == [7.0, 10.0, 12.0]
    assert text["moe_layer_list"] == [1, 2]
    assert text["moe_layers_enum"] == "1,2"
    assert config["quantization_config"]["modules_to_not_convert"] == [
        "model.layers.2.moe.experts.w13_weight",
        "vision_model.transformer.resblocks.0.attn.qkv_proj",
        "lm_head",
    ]


@pytest.mark.parametrize("layers", [[], [3, 0], [0, 0], [0, 6], [-1, 2]])
def test_plan_rejects_invalid_layers(layers):
    with pytest.raises(ValueError):
        slicer.plan_slice(_config(), layers)


def test_plan_rejects_inconsistent_moe_lists():
    config = _config()
    config["text_config"]["moe_layers_enum"] = "3,4"
    with pytest.raises(ValueError, match="different layers"):
        slicer.plan_slice(config, [0, 3])
