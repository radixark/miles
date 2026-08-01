from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
    disabled="FIXME: re-enable after the MXFP8 H200 reference path is settled.",
)


import json

import pytest
import safetensors.torch
import torch
from tools.convert_hf_to_mxfp8 import _add_dspark_stage_aliases, convert_mxfp8
from tools.convert_hf_to_mxfp8 import quantize_mxfp8 as tool_quantize_mxfp8
from tools.convert_hf_to_mxfp8 import should_quantize as tool_should_quantize_mxfp8
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.constants import TE_DType

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import (
    _quantize_param as processor_quantize_mxfp8_param,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import quantize_params_mxfp8

MXFP8_GROUP_SIZE = 32
MXFP8_SHAPES = [
    (1, 64),
    (1, 1024),
    (3, 128),
    (16, 64),
    (64, 128),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 1024),
    (1024, 2048),
    (7168, 2048),
    (2048, 7168),
    (128, 16384),
]


def _make_weight(init_data: str, dtype: torch.dtype, shape: tuple[int, int], device: str) -> torch.Tensor:
    m, n = shape
    if init_data == "random":
        return 16 * torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        base = torch.linspace(-512.0, 512.0, steps=n // 2, dtype=torch.float32, device=device)
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
        row = torch.empty(n, dtype=torch.float32, device=device)
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros((m, n), dtype=dtype, device=device)
    if init_data == "maxes":
        return torch.full((m, n), torch.finfo(dtype).max, dtype=dtype, device=device)
    raise ValueError(f"Unknown init_data: {init_data}")


def _processor_quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = dict(processor_quantize_mxfp8_param("model.layers.0.mlp.experts.0.down_proj.weight", weight))
    return (
        quantized["model.layers.0.mlp.experts.0.down_proj.weight"],
        quantized["model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"],
    )


def _te_mxfp8_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    m, k = weight.shape
    if m % MXFP8_GROUP_SIZE != 0:
        padded_m = ((m + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE) * MXFP8_GROUP_SIZE
        padded_weight = torch.zeros((padded_m, k), dtype=weight.dtype, device=weight.device)
        padded_weight[:m].copy_(weight)
    else:
        padded_weight = weight

    quantizer = MXFP8Quantizer(fp8_dtype=TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=False)
    quantized = quantizer(padded_weight)
    return (
        quantized._rowwise_data[:m].contiguous(),
        quantized._rowwise_scale_inv[:m, : k // MXFP8_GROUP_SIZE].contiguous(),
    )


def test_mxfp8_quantize_params_respects_extra_high_precision_layers_megatron():
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc2",)})()

    out = quantize_params_mxfp8(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp8"},
    )

    assert out is converted_named_params


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_mxfp8_quantize_params_respects_first_last_layers_bf16(layer_idx):
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", weight),
    ]
    args = type(
        "Args",
        (),
        {
            "first_last_layers_bf16": True,
            "num_layers": 4,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        },
    )()

    out = quantize_params_mxfp8(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "mxfp8"},
    )

    assert out is converted_named_params


def test_mxfp8_hf_should_quantize_respects_extra_high_precision_layers_hf():
    weight = torch.randn((4, MXFP8_GROUP_SIZE), dtype=torch.bfloat16)

    assert not tool_should_quantize_mxfp8(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.0",),
    )
    assert tool_should_quantize_mxfp8(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.1",),
    )


def test_mxfp8_hf_config_adds_dspark_stage_aliases_for_skipped_mtp_modules():
    mtp_modules = {
        "mtp.0.self_attn.wq_a",
        "mtp.0.self_attn.wkv",
        "mtp.0.main_proj",
        "mtp.0.mlp.shared_experts.gate_proj",
    }

    augmented = _add_dspark_stage_aliases(mtp_modules)

    assert augmented == mtp_modules | {
        "stages.0.self_attn.wq_a",
        "stages.0.self_attn.wkv",
        "mtp.0.self_attn.wqkv_a",
        "stages.0.self_attn.wqkv_a",
        "stages.0.main_proj",
        "stages.0.mlp.shared_experts.gate_proj",
    }
    assert not any(".mlp.experts" in name for name in augmented)


def test_mxfp8_hf_config_adds_fused_wqkv_a_alias_only_for_complete_pair():
    augmented = _add_dspark_stage_aliases(
        {
            "model.layers.0.self_attn.wq_a",
            "model.layers.0.self_attn.wkv",
            "model.layers.1.self_attn.wq_a",
        }
    )

    assert "model.layers.0.self_attn.wqkv_a" in augmented
    assert "model.layers.1.self_attn.wqkv_a" not in augmented


def test_mxfp8_hf_converter_propagates_nested_dspark_byte_for_byte(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    draft_dir = model_dir / "dspark"
    draft_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')
    root_shard = "model.safetensors"
    root_weights = {"model.layers.0.input_layernorm.weight": torch.ones(32, dtype=torch.bfloat16)}
    safetensors.torch.save_file(root_weights, model_dir / root_shard, metadata={"format": "pt"})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: root_shard for name in root_weights}})
    )

    draft_shard = "model-00002-of-00002.safetensors"
    draft_weights = {"mtp.0.ffn.experts.0.w1.weight": torch.arange(32, dtype=torch.int8).reshape(2, 16)}
    safetensors.torch.save_file(draft_weights, draft_dir / draft_shard, metadata={"format": "pt"})
    (draft_dir / "config.json").write_bytes(b'{"native_dspark":true}\n')
    (draft_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: draft_shard for name in draft_weights}})
    )
    draft_bytes = {path.name: path.read_bytes() for path in draft_dir.iterdir()}

    convert_mxfp8(str(model_dir), str(save_dir), device="cpu")

    assert {path.name: path.read_bytes() for path in (save_dir / "dspark").iterdir()} == draft_bytes
    output_index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    assert not any(name.startswith("mtp.") for name in output_index["weight_map"])
    output_config = json.loads((save_dir / "config.json").read_text())
    assert not any(
        name.startswith(("mtp.", "stages."))
        for name in output_config["quantization_config"].get("modules_to_not_convert", [])
    )


@pytest.mark.parametrize(
    "quantize_fn",
    [_processor_quantize_mxfp8, tool_quantize_mxfp8],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", MXFP8_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_mxfp8_quantize_matches_reference(quantize_fn, shape, dtype, init_data):
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, scale = quantize_fn(weight)
    qweight_ref, scale_ref = _te_mxfp8_reference(weight)

    assert qweight.shape == weight.shape
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (*weight.shape[:-1], weight.shape[-1] // MXFP8_GROUP_SIZE)
    assert scale.dtype == torch.uint8
    torch.testing.assert_close(qweight.view(dtype=torch.uint8), qweight_ref.view(dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
