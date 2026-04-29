from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-fast-1-gpu", num_gpus=1)


import pytest
import torch
from tools.convert_hf_to_mxfp8 import quantize_mxfp8 as tool_quantize_mxfp8

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8 import (
    _quantize_param as processor_quantize_mxfp8_param,
)

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


def _dequantize_mxfp8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale_fp32 = (scale.to(torch.int32) << 23).view(torch.float32)
    scale_fp32 = scale_fp32.repeat_interleave(MXFP8_GROUP_SIZE, dim=-1)
    return qweight.to(torch.float32) * scale_fp32


def _assert_mxfp8_dequant_close(actual: torch.Tensor, expected: torch.Tensor, init_data: str) -> None:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    assert not torch.any(torch.isnan(actual))
    assert not torch.any(torch.isnan(expected))
    assert actual.shape == expected.shape

    if init_data == "maxes":
        return

    assert not torch.any(torch.isinf(actual))
    left = torch.abs(actual - expected)
    right = 8.0 + 0.125 * torch.abs(expected)
    mismatch_percent = torch.sum(left > right) / actual.numel()
    assert mismatch_percent <= 0.001


@pytest.mark.parametrize(
    "quantize_fn",
    [_processor_quantize_mxfp8, tool_quantize_mxfp8],
    ids=["processor", "convert_tool"],
)
@pytest.mark.parametrize("shape", MXFP8_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
def test_mxfp8_quantize_dequant_matches_input(quantize_fn, shape, dtype, init_data):
    device = "cuda"
    torch.manual_seed(42)

    weight = _make_weight(init_data, dtype, shape, device)
    qweight, scale = quantize_fn(weight)

    assert qweight.shape == weight.shape
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (*weight.shape[:-1], weight.shape[-1] // MXFP8_GROUP_SIZE)
    assert scale.dtype == torch.uint8

    dequant = _dequantize_mxfp8(qweight, scale)
    _assert_mxfp8_dequant_close(dequant, weight, init_data)
