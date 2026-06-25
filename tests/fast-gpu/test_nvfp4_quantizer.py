from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])


import os

import pytest
import safetensors
import safetensors.torch
import torch
from tools.convert_hf_to_nvfp4 import convert_nvfp4
from tools.convert_hf_to_nvfp4 import quantize_nvfp4 as tool_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import should_quantize as tool_should_quantize_nvfp4
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    quantize_nvfp4 as processor_quantize_nvfp4,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.utils.nvfp4 import (
    NVFP4_GROUP_SIZE,
    nvfp4_global_decode_scale_te,
    nvfp4_quantize_1d_pair,
    nvfp4_weight_e4m3_max,
)

NVFP4_SHAPES = [
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
        return torch.randn((m, n), dtype=dtype, device=device)
    if init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device=device)
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


def _te_nvfp4_reference(
    weight: torch.Tensor,
    global_amax: torch.Tensor,
    row_scaled_nvfp4: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()
    qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
        weight,
        global_amax,
        NVFP4_GROUP_SIZE,
        1,
        pow_2_scales=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all"),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        eps=0.0,
    )
    return qweight, block_scale, nvfp4_global_decode_scale_te(global_amax, nvfp4_e4m3_max)


def test_nvfp4_quantize_params_requires_complete_gated_pair():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires gate/up tensors to be quantized together"):
        quantize_params_nvfp4(
            args=None,
            megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            converted_named_params=[
                ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
            ],
            quantization_config={"quant_method": "nvfp4"},
        )


def test_nvfp4_quantize_params_respects_extra_high_precision_layers_megatron():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
    ]
    args = type("Args", (), {"extra_high_precision_layers_megatron": ("linear_fc1",)})()

    out = quantize_params_nvfp4(
        args=args,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_nvfp4_quantize_params_respects_first_last_layers_bf16(layer_idx):
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    converted_named_params = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
        ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
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

    out = quantize_params_nvfp4(
        args=args,
        megatron_name=f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight0",
        converted_named_params=converted_named_params,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert out is converted_named_params


def test_nvfp4_quantize_params_omits_static_input_scale(monkeypatch):
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
    qweight = torch.empty((4, NVFP4_GROUP_SIZE // 2), dtype=torch.uint8)
    block_scale = torch.empty((4, 1), dtype=torch.float8_e4m3fn)
    global_scale = torch.ones((), dtype=torch.float32)

    def fake_quantize_1d_pair(_gate, _up):
        return (qweight, block_scale, global_scale), (qweight, block_scale, global_scale)

    monkeypatch.setattr(
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4.nvfp4_quantize_1d_pair",
        fake_quantize_1d_pair,
    )

    out = quantize_params_nvfp4(
        args=None,
        megatron_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        converted_named_params=[
            ("model.layers.0.mlp.experts.0.gate_proj.weight", weight),
            ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
        ],
        quantization_config={"quant_method": "nvfp4"},
    )

    names = [name for name, _ in out]
    assert "model.layers.0.mlp.experts.0.gate_proj.input_scale" not in names
    assert "model.layers.0.mlp.experts.0.up_proj.input_scale" not in names


def test_nvfp4_hf_should_quantize_respects_extra_high_precision_layers_hf():
    weight = torch.randn((4, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)

    assert not tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.0",),
    )
    assert tool_should_quantize_nvfp4(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        weight,
        skip_weight_substrings=("mlp.experts.1",),
    )


def test_nvfp4_hf_converter_quantizes_cross_shard_gated_pair_together(tmp_path, monkeypatch):
    monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')

    gate_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    up_key = "model.layers.0.mlp.experts.0.up_proj.weight"
    gate = torch.randn((3, 128), dtype=torch.bfloat16)
    up = torch.randn((5, 128), dtype=torch.bfloat16)
    safetensors.torch.save_file({gate_key: gate}, model_dir / "gate.safetensors", metadata={"format": "pt"})
    safetensors.torch.save_file({up_key: up}, model_dir / "up.safetensors", metadata={"format": "pt"})

    convert_nvfp4(str(model_dir), str(save_dir), device="cuda")

    (gate_qweight, gate_block_scale, gate_global_scale), (
        up_qweight,
        up_block_scale,
        up_global_scale,
    ) = nvfp4_quantize_1d_pair(gate.cuda(), up.cuda())

    with safetensors.safe_open(save_dir / "gate.safetensors", framework="pt", device="cuda") as f:
        torch.testing.assert_close(f.get_tensor(gate_key), gate_qweight, rtol=0, atol=0)
        torch.testing.assert_close(
            f.get_tensor(gate_key.replace(".weight", ".weight_scale")).view(torch.uint8),
            gate_block_scale.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            f.get_tensor(gate_key.replace(".weight", ".weight_scale_2")),
            gate_global_scale,
            rtol=0,
            atol=0,
        )

    with safetensors.safe_open(save_dir / "up.safetensors", framework="pt", device="cuda") as f:
        torch.testing.assert_close(f.get_tensor(up_key), up_qweight, rtol=0, atol=0)
        torch.testing.assert_close(
            f.get_tensor(up_key.replace(".weight", ".weight_scale")).view(torch.uint8),
            up_block_scale.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            f.get_tensor(up_key.replace(".weight", ".weight_scale_2")),
            up_global_scale,
            rtol=0,
            atol=0,
        )


def test_nvfp4_hf_converter_quantizes_same_shard_gated_pair_together(tmp_path, monkeypatch):
    monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')

    gate_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    up_key = "model.layers.0.mlp.experts.0.up_proj.weight"
    gate = torch.randn((3, 128), dtype=torch.bfloat16)
    up = torch.randn((5, 128), dtype=torch.bfloat16)
    safetensors.torch.save_file(
        {
            gate_key: gate,
            up_key: up,
        },
        model_dir / "model.safetensors",
        metadata={"format": "pt"},
    )

    convert_nvfp4(str(model_dir), str(save_dir), device="cuda")

    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cuda") as f:
        gate_global_scale = f.get_tensor(gate_key.replace(".weight", ".weight_scale_2"))
        up_global_scale = f.get_tensor(up_key.replace(".weight", ".weight_scale_2"))
        torch.testing.assert_close(gate_global_scale, up_global_scale, rtol=0, atol=0)


def test_nvfp4_quantize_pair_reuses_adjacent_storage(monkeypatch):
    base = torch.randn((32, 64), dtype=torch.bfloat16, device="cuda")
    gate, up = base.chunk(2, dim=0)

    def fail_cat(*args, **kwargs):
        raise AssertionError("adjacent gate/up pair should not be materialized with torch.cat")

    monkeypatch.setattr(torch, "cat", fail_cat)
    (gate_qweight, gate_block_scale, _), (up_qweight, up_block_scale, _) = nvfp4_quantize_1d_pair(gate, up)

    assert gate_qweight.shape == (16, 32)
    assert up_qweight.shape == (16, 32)
    assert gate_block_scale.shape == (16, 4)
    assert up_block_scale.shape == (16, 4)


@pytest.mark.parametrize(
    "quantize_fn",
    [processor_quantize_nvfp4, tool_quantize_nvfp4],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("use_4over6", [False, True], ids=["default", "4over6"])
def test_nvfp4_quantize_matches_te_reference_bitwise(quantize_fn, shape, dtype, init_data, use_4over6, monkeypatch):
    device = "cuda"
    torch.manual_seed(42)
    if use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    else:
        monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)

    weight = _make_weight(init_data, dtype, shape, device)
    reference_amax = torch.max(torch.abs(weight.to(torch.float32)))
    qweight, block_scale, global_scale = quantize_fn(weight)
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference(
        weight,
        reference_amax,
        row_scaled_nvfp4=False,
    )

    torch.testing.assert_close(qweight, qweight_ref, rtol=0, atol=0)
    torch.testing.assert_close(block_scale.view(torch.uint8), block_scale_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(global_scale, global_scale_ref, rtol=0, atol=0)


@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("use_4over6", [False, True], ids=["default", "4over6"])
def test_nvfp4_quantize_pair_matches_te_reference_bitwise(shape, dtype, init_data, use_4over6, monkeypatch):
    device = "cuda"
    torch.manual_seed(42)
    if use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    else:
        monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)

    gate = _make_weight(init_data, dtype, shape, device)
    up = _make_weight(init_data, dtype, shape, device)
    (gate_qweight, gate_block_scale, gate_global_scale), (
        up_qweight,
        up_block_scale,
        up_global_scale,
    ) = nvfp4_quantize_1d_pair(gate, up)

    combined = torch.cat((gate, up), dim=0)
    qweight_ref, block_scale_ref, global_scale_ref = _te_nvfp4_reference(
        combined,
        torch.max(torch.abs(combined.to(torch.float32))),
        row_scaled_nvfp4=False,
    )

    torch.testing.assert_close(gate_qweight, qweight_ref[: gate.shape[0]], rtol=0, atol=0)
    torch.testing.assert_close(up_qweight, qweight_ref[gate.shape[0] :], rtol=0, atol=0)
    torch.testing.assert_close(
        gate_block_scale.view(torch.uint8), block_scale_ref[: gate.shape[0]].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        up_block_scale.view(torch.uint8), block_scale_ref[gate.shape[0] :].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(gate_global_scale, global_scale_ref, rtol=0, atol=0)
    torch.testing.assert_close(up_global_scale, global_scale_ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
