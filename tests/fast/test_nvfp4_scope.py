import json

import pytest
import safetensors
import safetensors.torch
import torch
from tools.convert_hf_to_nvfp4 import convert_nvfp4, should_quantize

from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_nvfp4
from miles.utils.nvfp4 import NVFP4_GROUP_SIZE


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", True),
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", True),
        ("model.layers.0.moe.experts.0.gate_up_proj.weight", True),
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", False),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", False),
        ("model.layers.0.mlp.shared_experts.experts.0.gate_proj.weight", False),
        ("model.layers.0.mlp.gate_proj.weight", False),
    ],
)
def test_nvfp4_hf_scope_targets_only_routed_experts(name, expected):
    weight = torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)

    assert should_quantize(name, weight) is expected


@pytest.mark.parametrize(
    "megatron_name",
    [
        "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
        "decoder.layers.0.mlp.shared_experts.linear_fc2.weight",
        "mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight",
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight",
    ],
)
def test_nvfp4_megatron_export_preserves_shared_experts(monkeypatch, megatron_name):
    converted_named_params = [
        (
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        )
    ]

    def fail_quantize(*_args, **_kwargs):
        raise AssertionError("shared experts must not enter NVFP4 quantization")

    monkeypatch.setattr(quantizer_nvfp4, "_quantize_moe_params", fail_quantize)

    result = quantizer_nvfp4.quantize_params_nvfp4(
        args=None,
        megatron_name=megatron_name,
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert result is converted_named_params


def test_nvfp4_megatron_export_still_targets_routed_experts(monkeypatch):
    converted_named_params = [
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        )
    ]
    quantized = [("quantized", torch.ones((), dtype=torch.float32))]

    monkeypatch.setattr(quantizer_nvfp4, "_quantize_moe_params", lambda *_args: quantized)

    result = quantizer_nvfp4.quantize_params_nvfp4(
        args=None,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert result is quantized


def test_nvfp4_hf_conversion_preserves_shared_expert_storage(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')

    weights = {
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        "model.layers.0.mlp.shared_experts.up_proj.weight": torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        "model.layers.0.mlp.shared_experts.down_proj.weight": torch.ones((2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        "model.layers.0.mlp.shared_experts.experts.0.gate_proj.weight": torch.ones(
            (2, NVFP4_GROUP_SIZE), dtype=torch.bfloat16
        ),
    }
    safetensors.torch.save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})

    convert_nvfp4(str(model_dir), str(save_dir), device="cpu")

    config = json.loads((save_dir / "config.json").read_text())
    ignored = config["quantization_config"]["ignore"]
    assert all(name.removesuffix(".weight") in ignored for name in weights)

    hf_quant_config = json.loads((save_dir / "hf_quant_config.json").read_text())
    assert hf_quant_config["quantization"]["exclude_modules"] == ignored

    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as f:
        assert set(f.keys()) == set(weights)
        assert all(f.get_tensor(name).dtype == torch.bfloat16 for name in weights)
