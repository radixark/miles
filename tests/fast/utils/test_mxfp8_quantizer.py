from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-fast-1-gpu", num_gpus=1)


import pytest
import torch
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
