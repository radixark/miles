from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-c-8-gpu-b200",
    labels=["precision"],
    hardware=["blackwell"],
)


import contextlib
import json
import os
import sys
from datetime import timedelta
from types import ModuleType

import pytest
import safetensors
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch as te
from tools.convert_hf_to_nvfp4 import convert_nvfp4
from tools.convert_hf_to_nvfp4 import quantize_nvfp4 as tool_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import should_quantize as tool_should_quantize_nvfp4
from torch.utils._python_dispatch import TorchDispatchMode

import miles.utils.fused_nvfp4_qdq as qdq_kernels
import miles.utils.nvfp4_fake_qat as nvfp4_qat
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    quantize_nvfp4 as processor_quantize_nvfp4,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_grouped_nvfp4_amax,
    compute_nvfp4_amax,
    current_nvfp4_qdq_config,
    fake_grouped_nvfp4_quantization_ste,
    fake_nvfp4_quantization_ste,
    fused_grouped_nvfp4_qdq,
    fused_nvfp4_qdq,
)
from miles.utils.nvfp4 import (
    NVFP4_GROUP_SIZE,
    nvfp4_global_decode_scale_te,
    nvfp4_global_encode_scale_te,
    nvfp4_quantize_1d_pair,
    nvfp4_weight_e4m3_max,
)

# Newer TE builds removed this legacy export oracle. Keep its cases isolated
# so native QDQ/STE tests still collect and run without it.
try:
    from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef
except ModuleNotFoundError as exc:
    if exc.name != "transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4":
        raise
    NVFP4QuantizerRef = None


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


class _NoHostScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, "Scale conversion read a device scalar"
        assert func != torch.ops.aten.lift_fresh.default, "Scale conversion created a host scalar tensor"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("e4m3_max", [256, 448])
@pytest.mark.parametrize("shape", [(), (1,), (8,), (2, 4)])
def test_nvfp4_global_scale_exact_without_host_scalar_read(device, e4m3_max, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Includes overflow clamping, underflow-to-zero repair and NaN propagation.
    values = torch.tensor(
        [
            0.0,
            1.0,
            3.5,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    # Frozen legacy batched arithmetic on CPU, independent of the device helper.
    numerator = torch.tensor(float(e4m3_max), dtype=torch.float32) * torch.tensor(6.0, dtype=torch.float32)
    expected = torch.div(numerator, values)
    expected = torch.min(expected, torch.tensor(torch.finfo(torch.float32).max))
    expected = torch.where(expected == 0.0, torch.ones_like(expected), expected)

    case_size = torch.Size(shape).numel()
    for cpu_amax, expected_encode in zip(values.split(case_size), expected.split(case_size), strict=True):
        amax = cpu_amax.reshape(shape).to(device)
        expected_encode = expected_encode.reshape(shape)
        with _NoHostScalarRead():
            encoded = nvfp4_global_encode_scale_te(amax, e4m3_max)
            decoded = nvfp4_global_decode_scale_te(amax, e4m3_max)
        assert encoded.device == amax.device and encoded.shape == amax.shape
        torch.testing.assert_close(encoded.cpu(), expected_encode, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(decoded.cpu(), torch.div(1.0, expected_encode), rtol=0, atol=0, equal_nan=True)


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


def test_nvfp4_hf_converter_uses_compact_bf16_moe_prefixes(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')

    weights = {
        f"model.layers.0.mlp.experts.{expert_idx}.{projection}.weight": torch.ones(
            (1, NVFP4_GROUP_SIZE), dtype=torch.bfloat16
        )
        for expert_idx in range(128)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    weights["model.layers.0.input_layernorm.weight"] = torch.ones(NVFP4_GROUP_SIZE, dtype=torch.bfloat16)
    safetensors.torch.save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})

    convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        num_layers_at_end_in_bf16=1,
    )

    expected_ignore = [
        "model.layers.0.",
        "model.layers.0.input_layernorm",
        "model.layers.0.mlp.experts",
    ]
    config = json.loads((save_dir / "config.json").read_text())
    assert config["quantization_config"]["ignore"] == expected_ignore

    hf_quant_config = json.loads((save_dir / "hf_quant_config.json").read_text())
    assert hf_quant_config["quantization"]["exclude_modules"] == expected_ignore

    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as f:
        assert all("weight_scale" not in key for key in f.keys())
        assert f.get_tensor("model.layers.0.mlp.experts.127.down_proj.weight").dtype == torch.bfloat16


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
@pytest.mark.skipif(NVFP4QuantizerRef is None, reason="TE legacy NVFP4 export oracle is unavailable")
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
@pytest.mark.skipif(NVFP4QuantizerRef is None, reason="TE legacy NVFP4 export oracle is unavailable")
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


# Data modes and the 4over6 matrix mirror FlashInfer
# tests/utils/test_fp4_quantize.py::test_nvfp4_quantize_te_reference. The
# per-tensor oracle follows Transformer Engine's strict
# tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py test and calls native
# quantize-dequantize because FlashInfer's per-tensor contract differs.
@pytest.fixture(scope="module", autouse=True)
def _select_local_cuda_device() -> None:
    """Keep torchrun workers on their assigned GPUs without initializing collectives."""
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


NVFP4_QDQ_SHAPES = [
    # Minimum-K and odd-row cases absent from FlashInfer's swizzled-layout matrix.
    (1, 16),
    (1, 32),
    (3, 48),
    # FlashInfer strict-test shapes and both TE BF16 dispatch routes after M padding.
    (1, 64),
    (3, 128),
    (16, 64),
    (31, 128),
    (32, 128),
    (128, 64),
    (128, 1024),
    (256, 256),
    (1024, 2048),
]


NVFP4_QDQ_CONFIGS = [pytest.param(NVFP4QDQConfig(), id="nvfp4")]
for _error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
    for _e4m3_max in (448, 256):
        for _error_use_fast_math in (False, True):
            NVFP4_QDQ_CONFIGS.append(
                pytest.param(
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=_e4m3_max,
                        error_mode=_error_mode,
                        error_use_fast_math=_error_use_fast_math,
                    ),
                    id=(
                        f"4over6-{_error_mode.name.lower()}-e4m3-{_e4m3_max}-"
                        f"{'fp16-error' if _error_use_fast_math else 'exact-error'}"
                    ),
                )
            )


def _make_qdq_input(shape: tuple[int, int], dtype: torch.dtype, init_data: str) -> torch.Tensor:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    m, n = shape
    if init_data == "random":
        x = torch.randn(shape, dtype=dtype, device="cuda")
        if m > 1:
            x[0].zero_()
        return x
    if init_data == "boundary":
        base = torch.linspace(-12.0, 12.0, steps=n // 2, dtype=torch.float32, device="cuda")
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, torch.full_like(base, 1e-4))
        row = torch.empty(n, dtype=torch.float32, device="cuda")
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        # Alternate signed zeros so the integer-view equality below exercises
        # TE's E2M1 sign-bit contract for zero-amax blocks.
        return torch.tensor([-0.0, 0.0], dtype=torch.float32, device="cuda").repeat(m, n // 2).to(dtype=dtype)
    if init_data == "maxes":
        return torch.full(shape, torch.finfo(dtype).max, dtype=dtype, device="cuda")
    raise ValueError(f"Unknown init_data: {init_data}")


def _make_te_qdq_quantizer(config: NVFP4QDQConfig):
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=config.use_4over6,
        nvfp4_e4m3_max=config.e4m3_max,
        nvfp4_4over6_err_mode=config.error_mode.name,
        with_random_sign_mask=False,
    )


def _te_qdq_reference(x: torch.Tensor, config: NVFP4QDQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    padded_m = ((m + 15) // 16) * 16
    if padded_m == m:
        x_padded = x.contiguous()
    else:
        padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x.contiguous(), padding), dim=0)

    quantized = _make_te_qdq_quantizer(config).quantize(x_padded)
    reference = quantized.dequantize(dtype=x.dtype)[:m, :n].contiguous()
    assert quantized._amax_rowwise is not None
    return reference, quantized._amax_rowwise.reshape(1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("shape", NVFP4_QDQ_SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}")
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", NVFP4_QDQ_CONFIGS)
@torch.inference_mode()
def test_fused_nvfp4_qdq_is_bit_exact_with_te(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    """Cover BF16/FP16 x shapes x data patterns x the full supported feature matrix."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fast_math else "0")
    x = _make_qdq_input(shape, dtype, init_data)
    amax = compute_nvfp4_amax(x)
    expected, te_amax = _te_qdq_reference(x, config)
    actual = fused_nvfp4_qdq(x, amax, config)

    assert torch.equal(amax.reshape(1).view(torch.int32), te_amax.view(torch.int32))
    # Integer views distinguish signed zero; tolerance-zero floating comparison does not.
    actual_bits = actual.view(torch.uint16)
    expected_bits = expected.view(torch.uint16)
    assert torch.equal(
        actual_bits, expected_bits
    ), f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fused_nvfp4_qdq_uses_straight_through_gradient_and_preserves_main_grad() -> None:
    x = torch.randn((3, 32), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    main_grad = torch.empty_like(x)
    x.main_grad = main_grad
    output = fake_nvfp4_quantization_ste(x, NVFP4QDQConfig())
    output.backward(torch.ones_like(output))

    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)
    assert output.main_grad is main_grad


@pytest.mark.parametrize(
    ("four_over_six_scope", "e4m3_256_scope", "expected_enabled", "expected_max"),
    [
        (
            four_over_six_scope,
            e4m3_256_scope,
            four_over_six_scope in ("weights", "all"),
            expected_max,
        )
        for four_over_six_scope in ("none", "activations", "weights", "all")
        for e4m3_256_scope in ("none", "activations", "weights", "all")
        for expected_max in [
            256 if four_over_six_scope in ("weights", "all") and e4m3_256_scope in ("weights", "all") else 448
        ]
    ],
)
@pytest.mark.parametrize(
    ("error_mode", "error_use_fast_math"), [("MAE", False), ("MAE", True), ("MSE", False), ("MSE", True)]
)
def test_current_nvfp4_qdq_config_maps_full_latest_te_env_contract(
    monkeypatch: pytest.MonkeyPatch,
    four_over_six_scope: str,
    e4m3_256_scope: str,
    expected_enabled: bool,
    expected_max: int,
    error_mode: str,
    error_use_fast_math: bool,
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", four_over_six_scope)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", e4m3_256_scope)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", error_mode)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if error_use_fast_math else "0")
    config = current_nvfp4_qdq_config()
    assert config.use_4over6 is expected_enabled
    assert config.e4m3_max == expected_max
    assert config.error_mode is NVFP4QDQErrorMode[error_mode]
    # The latest TE meaning is FP16-rounded candidate error, not a general
    # instruction-level fast-math toggle.
    assert config.error_use_fast_math is (expected_enabled and error_use_fast_math)


@pytest.mark.parametrize("legacy_scope", ["inputs", "gradients"])
def test_current_nvfp4_qdq_config_rejects_stale_te_scopes(monkeypatch: pytest.MonkeyPatch, legacy_scope: str) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", legacy_scope)
    with pytest.raises(ValueError, match="activations"):
        current_nvfp4_qdq_config()


def test_current_nvfp4_qdq_config_rejects_quant_fast_math(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    with pytest.raises(ValueError, match="NVTE_USE_FAST_MATH=0"):
        current_nvfp4_qdq_config()


@pytest.mark.parametrize(
    "flag_name",
    ["NVTE_USE_FAST_MATH", "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"],
)
def test_current_nvfp4_qdq_config_rejects_non_numeric_bool_flags(
    monkeypatch: pytest.MonkeyPatch, flag_name: str
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "none")
    monkeypatch.setenv(flag_name, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        current_nvfp4_qdq_config()


def test_te_grouped_linear_real_discrete_weight_qdq_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.extensions import transformer_engine as te_extension
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    group_count, rows, columns = 3, 2, 32
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "none")
    monkeypatch.delenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", raising=False)
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=columns,
        num_attention_heads=1,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        moe_single_grouped_weight=False,
    )
    pg_collection = ProcessGroupCollection()
    pg_collection.expt_tp = None
    layer = te_extension.TEGroupedLinear(
        num_gemms=group_count,
        input_size=columns,
        output_size=rows,
        parallel_mode=None,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        is_expert=True,
        pg_collection=pg_collection,
    )

    assert layer.fuse_wgrad_accumulation is False
    assert not getattr(layer, "single_grouped_weight", False)
    assert getattr(layer, "weight", None) is None
    weights = [getattr(layer, f"weight{group_idx}") for group_idx in range(group_count)]
    assert set(dict(layer.named_parameters())) == {f"weight{group_idx}" for group_idx in range(group_count)}

    actual_weights = te_extension.TEGroupedLinear._get_weight_tensors(layer)
    qdq_config = current_nvfp4_qdq_config()
    expected_weights = [_te_qdq_reference(weight, qdq_config)[0] for weight in weights]

    assert len(actual_weights) == group_count
    assert all(weight.requires_grad for weight in actual_weights)
    assert all(
        torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
        for actual, expected in zip(actual_weights, expected_weights, strict=False)
    )

    m_splits = [2, 1, 3]
    inp = torch.ones((sum(m_splits), columns), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    output, bias = layer(inp, m_splits)
    output.backward(torch.ones_like(output))

    assert bias is None
    assert tuple(output.shape) == (sum(m_splits), rows)
    assert inp.grad is not None and torch.isfinite(inp.grad).all()
    for weight, m_split in zip(weights, m_splits, strict=False):
        assert weight.grad is not None
        torch.testing.assert_close(weight.grad, torch.full_like(weight.grad, m_split), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_nvfp4_qdq_rejects_unsupported_input_dtype(dtype: torch.dtype) -> None:
    x = torch.randn((2, 16), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="supports BF16 and FP16"):
        fused_nvfp4_qdq(x, x.abs().amax().float(), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_non_block_aligned_k() -> None:
    x = torch.randn((2, 17), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="K divisible by 16"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_misaligned_contiguous_storage() -> None:
    storage = torch.randn(33, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(2, 16)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte-aligned"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_nvfp4_qdq_uses_and_restores_non_current_device() -> None:
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        pytest.skip("Run the cross-device state test in a dedicated single process")
    primary_device = torch.cuda.current_device()
    secondary_device = (primary_device + 1) % torch.cuda.device_count()
    with torch.cuda.device(primary_device):
        with torch.cuda.device(secondary_device):
            x = _make_qdq_input((3, 32), torch.bfloat16, "boundary")
            amax = compute_nvfp4_amax(x)
            expected, _ = _te_qdq_reference(x, NVFP4QDQConfig())

        assert torch.cuda.current_device() == primary_device
        actual = fused_nvfp4_qdq(x, amax, NVFP4QDQConfig())
        assert torch.cuda.current_device() == primary_device

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


class TestNVFP4FakeQATAdapter:
    def test_disabled_path_returns_original_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = [torch.nn.Parameter(torch.empty(4, 16))]
        monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "0")

        actual = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(weights)

        assert actual is weights

    def test_enabled_path_resolves_config_once_and_maps_arbitrary_weight_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        weights = [torch.nn.Parameter(torch.empty(4, 16)) for _ in range(3)]
        expected = [torch.empty_like(weight) for weight in weights]
        qdq_config = object()
        config_calls = 0
        calls = []
        fake_module = ModuleType("miles.utils.fused_nvfp4_qdq")

        def current_config():
            nonlocal config_calls
            config_calls += 1
            return qdq_config

        def fake_qdq(weight, config):
            calls.append((weight, config))
            return expected[len(calls) - 1]

        fake_module.current_nvfp4_qdq_config = current_config
        fake_module.fake_nvfp4_quantization_ste = fake_qdq
        monkeypatch.setitem(sys.modules, "miles.utils.fused_nvfp4_qdq", fake_module)
        monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "1")

        actual = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(weights)

        assert config_calls == 1
        assert len(actual) == len(weights)
        assert all(value is expected_value for value, expected_value in zip(actual, expected, strict=False))
        assert all(weight is call[0] for weight, call in zip(weights, calls, strict=False))
        assert all(call[1] is qdq_config for call in calls)


def _make_grouped_qdq_input(
    group_count: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    init_data: str,
) -> torch.Tensor:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m, n = shape
    if init_data == "random":
        x = torch.randn((group_count, m, n), dtype=torch.float32, device="cuda")
        if m > 1:
            x[:, 0].zero_()
        scales = torch.pow(
            2.0,
            (torch.arange(group_count, device="cuda") % 5) - 2,
        ).view(-1, 1, 1)
        return (x * scales).to(dtype)
    if init_data == "boundary":
        weight = _make_qdq_input(shape, torch.float32, init_data)
        groups = []
        for group_idx in range(group_count):
            scale = 2.0 ** ((group_idx % 5) - 2)
            groups.append(torch.roll(weight, shifts=2 * group_idx, dims=1) * scale)
        return torch.stack(groups).to(dtype=dtype)
    if init_data == "zeros":
        weight = _make_qdq_input(shape, dtype, init_data)
        return torch.stack([torch.roll(weight, shifts=group_idx % 2, dims=1) for group_idx in range(group_count)])
    if init_data == "maxes":
        x = torch.full((group_count, m, n), torch.finfo(dtype).max, dtype=dtype, device="cuda")
        signs = torch.where(
            torch.arange(group_count, device="cuda") % 2 == 0,
            1.0,
            -1.0,
        ).to(dtype)
        return x * signs.view(-1, 1, 1)
    raise ValueError(f"Unknown init_data: {init_data}")


def _te_grouped_qdq_reference(x: torch.Tensor, config: NVFP4QDQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    references, amaxes = zip(*[_te_qdq_reference(weight, config) for weight in x.unbind(0)], strict=True)
    return torch.stack(references), torch.cat(amaxes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("group_count", [1, 3, 8], ids=lambda value: f"g{value}")
@pytest.mark.parametrize("shape", NVFP4_QDQ_SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}")
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", NVFP4_QDQ_CONFIGS)
@torch.inference_mode()
def test_grouped_nvfp4_qdq_is_bit_exact_with_te(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    group_count: int,
    shape: tuple[int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    """Cover BF16/FP16 x shapes x data patterns x the full supported feature matrix."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fast_math else "0")
    x = _make_grouped_qdq_input(group_count, shape, dtype, init_data)
    amaxes = compute_grouped_nvfp4_amax(x)
    expected, te_amax = _te_grouped_qdq_reference(x, config)
    actual = fused_grouped_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amax.view(torch.int32))
    # Integer views distinguish signed zero; tolerance-zero floating comparison does not.
    actual_bits = actual.view(torch.uint16)
    expected_bits = expected.view(torch.uint16)
    assert torch.equal(
        actual_bits, expected_bits
    ), f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    # Also protect the existing rank-2 path when sharing the store helper.
    loop = torch.stack([fused_nvfp4_qdq(w, compute_nvfp4_amax(w), config) for w in x.unbind(0)])
    assert torch.equal(actual_bits, loop.view(torch.uint16))


@pytest.mark.parametrize("group_count", [1, 8], ids=lambda value: f"g{value}")
def test_grouped_nvfp4_qdq_uses_straight_through_gradient_and_preserves_main_grad(
    group_count: int,
) -> None:
    x = torch.nn.Parameter(torch.randn((group_count, 3, 32), dtype=torch.bfloat16, device="cuda"))
    main_grad = torch.empty_like(x)
    x.main_grad = main_grad
    output = fake_grouped_nvfp4_quantization_ste(x, NVFP4QDQConfig())
    grad = (
        torch.arange(1, group_count + 1, dtype=torch.float32, device="cuda")
        .view(-1, 1, 1)
        .expand_as(output)
        .to(output.dtype)
    )
    output.backward(grad)

    assert tuple(output.shape) == tuple(x.shape)
    assert output.is_contiguous()
    torch.testing.assert_close(x.grad, grad, rtol=0.0, atol=0.0)
    assert output.main_grad is main_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_nvfp4_qdq_rejects_unsupported_input_dtype(dtype: torch.dtype) -> None:
    x = torch.randn((1, 2, 16), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="supports BF16 and FP16"):
        fused_grouped_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


def test_grouped_nvfp4_qdq_rejects_rank_2_input() -> None:
    x = torch.randn((2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="rank-3"):
        fused_grouped_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("group_count", [0, 2049])
def test_grouped_nvfp4_qdq_rejects_group_count_outside_bound(
    group_count: int,
) -> None:
    x = torch.empty((group_count, 1, 16), dtype=torch.bfloat16, device="cuda")
    amaxes = torch.empty(group_count, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="1 <= G <= 2048"):
        fused_grouped_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())


def test_grouped_nvfp4_qdq_rejects_non_block_aligned_k() -> None:
    x = torch.randn((3, 2, 17), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="N divisible by 16"):
        fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), NVFP4QDQConfig())


def test_grouped_nvfp4_qdq_rejects_misaligned_contiguous_storage() -> None:
    storage = torch.randn(33, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(1, 2, 16)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte-aligned"):
        fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), NVFP4QDQConfig())


def test_grouped_nvfp4_qdq_rejects_wrong_amax_shape() -> None:
    x = torch.randn((3, 2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        fused_grouped_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"), NVFP4QDQConfig())


@torch.inference_mode()
def test_grouped_nvfp4_qdq_supports_maximum_group_count() -> None:
    config = NVFP4QDQConfig()
    x = _make_grouped_qdq_input(2048, (1, 16), torch.bfloat16, "random")
    amaxes = compute_grouped_nvfp4_amax(x)
    expected, te_amaxes = _te_grouped_qdq_reference(x, config)
    actual = fused_grouped_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amaxes.view(torch.int32))
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_grouped_nvfp4_qdq_uses_and_restores_non_current_device() -> None:
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        pytest.skip("Run the cross-device state test in a dedicated single process")
    primary_device = torch.cuda.current_device()
    secondary_device = (primary_device + 1) % torch.cuda.device_count()
    with torch.cuda.device(primary_device):
        with torch.cuda.device(secondary_device):
            x = _make_grouped_qdq_input(3, (3, 32), torch.bfloat16, "boundary")
            amaxes = compute_grouped_nvfp4_amax(x)
            expected, _ = _te_grouped_qdq_reference(x, NVFP4QDQConfig())

        assert torch.cuda.current_device() == primary_device
        actual = fused_grouped_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())
        assert torch.cuda.current_device() == primary_device

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


@pytest.fixture
def grouped_qat_env(monkeypatch):
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "none")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")


def _native_grouped_layer(fuse_wgrad=False):
    import inspect

    if "use_grouped_tensor" not in inspect.signature(te.GroupedLinear).parameters:
        pytest.skip("Native packed TE GroupedLinear is required")

    class QATGroupedLinear(te.GroupedLinear):
        # Same hook used by the Miles Megatron fork.
        def _get_weight_tensors(self):
            return nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(super()._get_weight_tensors())

    return QATGroupedLinear(
        3,
        128,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        single_grouped_weight=True,
        use_grouped_tensor=True,
        fuse_wgrad_accumulation=fuse_wgrad,
    )


@pytest.mark.usefixtures("grouped_qat_env")
@pytest.mark.parametrize("use_4over6", [False, True])
@pytest.mark.parametrize("fuse_wgrad", [False, True])
def test_grouped_native_te_forward_backward_and_update(monkeypatch, use_4over6, fuse_wgrad):
    if use_4over6:
        monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
        monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    layer = _native_grouped_layer(fuse_wgrad)
    weight = layer.weight
    assert list(dict(layer.named_parameters())) == ["weight"]
    original = weight.rowwise_data.view(3, 64, 128).clone()
    original_ptr = weight.rowwise_data.data_ptr()
    if fuse_wgrad:
        weight.main_grad = torch.zeros_like(original, dtype=torch.float32)
    qweight = layer._get_weight_tensors()[0]
    assert qweight.requires_grad and qweight.grad_fn is not None
    assert qweight.rowwise_data.data_ptr() != original_ptr
    if fuse_wgrad:
        assert qweight.main_grad is weight.main_grad
    expected, _ = _te_grouped_qdq_reference(original, current_nvfp4_qdq_config())
    assert torch.equal(qweight.rowwise_data.view(3, 64, 128).view(torch.uint16), expected.view(torch.uint16))
    assert torch.equal(original, weight.rowwise_data.view_as(original))

    # Include a zero-token expert and different splits to catch expert reordering.
    splits = [2, 0, 3]
    inputs = torch.cat(
        [torch.full((tokens, 128), i + 1, device="cuda", dtype=torch.bfloat16) for i, tokens in enumerate(splits)]
    ).requires_grad_()
    output = layer(inputs, torch.tensor(splits, device="cuda", dtype=torch.int64))
    reference = torch.cat(
        [torch.nn.functional.linear(inp, w) for inp, w in zip(inputs.split(splits), expected, strict=True)]
    )
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.01)
    output.backward(torch.ones_like(output))
    expected_grad = torch.stack([torch.full_like(original[0], n * (i + 1)) for i, n in enumerate(splits)])
    grad = weight.main_grad if fuse_wgrad else weight.grad
    torch.testing.assert_close(grad, expected_grad.to(grad.dtype), rtol=0, atol=0)
    expected_dgrad = torch.cat(
        [w.float().sum(0).to(inputs.dtype).expand(n, -1) for w, n in zip(expected, splits, strict=True)]
    )
    torch.testing.assert_close(inputs.grad, expected_dgrad, rtol=0.02, atol=0.01)

    # Fused accumulation writes main_grad; the optimizer consumes that buffer.
    if fuse_wgrad:
        weight.grad = weight.main_grad.to(weight.dtype)
    optimizer = torch.optim.SGD([weight], lr=0.125)
    optimizer.step()
    assert layer.weight is weight and weight.rowwise_data.data_ptr() == original_ptr
    torch.testing.assert_close(weight.rowwise_data.view_as(original), original - expected_grad * 0.125, rtol=0, atol=0)


@pytest.mark.usefixtures("grouped_qat_env")
@pytest.mark.parametrize("native", [False, True])
def test_grouped_graph_replay_recomputes_each_expert_amax(native):
    if native:
        weight = _native_grouped_layer().weight
        storage = weight.rowwise_data.view(3, 64, 128)
    else:
        weight = torch.nn.Parameter(torch.randn((3, 64, 128), device="cuda", dtype=torch.bfloat16))
        storage = weight.detach()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors([weight])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors([weight])[0]
    result = output.rowwise_data.view_as(storage) if native else output
    for scale in (0.01, 100.0):
        with torch.no_grad():
            storage[1].normal_().mul_(scale)
            storage[2].zero_()
        graph.replay()
        expected, _ = _te_grouped_qdq_reference(storage, NVFP4QDQConfig())
        assert torch.equal(result.view(torch.uint16), expected.view(torch.uint16))


@pytest.mark.usefixtures("grouped_qat_env")
def test_grouped_native_adapter_rejects_irregular_layout():
    weight = _native_grouped_layer().weight
    weight.offsets = [0, 0, 0]
    with pytest.raises(ValueError, match="uniform, unquantized, densely ordered"):
        nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors([weight])


@pytest.mark.usefixtures("grouped_qat_env")
def test_grouped_native_checkpoint_keeps_original_high_precision_weights(tmp_path):
    layer = _native_grouped_layer()
    original = layer.weight.rowwise_data.clone()
    layer._get_weight_tensors()
    checkpoint = tmp_path / "grouped.pt"
    torch.save(layer.state_dict(), checkpoint)
    restored = _native_grouped_layer()
    with torch.serialization.safe_globals([type(layer.weight)]):
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(restored.weight.rowwise_data, original)
    actual = restored._get_weight_tensors()[0].rowwise_data
    expected = layer._get_weight_tensors()[0].rowwise_data
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


def test_grouped_rejects_noncontiguous_input_and_amax():
    x = torch.randn((3, 32, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        fused_grouped_nvfp4_qdq(x.transpose(1, 2), compute_grouped_nvfp4_amax(x), NVFP4QDQConfig())
    with pytest.raises(ValueError, match="contiguous"):
        fused_grouped_nvfp4_qdq(x, torch.ones(6, device="cuda")[::2], NVFP4QDQConfig())


@pytest.mark.usefixtures("grouped_qat_env")
@pytest.mark.parametrize("overwrite", [False, True])
def test_grouped_native_fused_wgrad_reaches_megatron_leaf_hook(overwrite):
    layer = _native_grouped_layer(fuse_wgrad=True)
    weight = layer.weight
    weight.main_grad = torch.ones((3, 64, 128), device="cuda", dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = True
    weight.overwrite_main_grad = overwrite
    weight.get_main_grad = lambda: weight.main_grad
    hook_calls = []

    def ddp_hook(param):
        # Megatron uses this marker to skip a second add into main_grad, and
        # the hook itself marks the parameter ready for gradient reduction.
        assert param.grad_added_to_main_grad
        assert torch.count_nonzero(param.grad).item() == 0
        hook_calls.append(True)
        param.grad = None

    weight.register_post_accumulate_grad_hook(ddp_hook)
    splits = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int64)
    inputs = torch.ones((5, 128), device="cuda", dtype=torch.bfloat16)
    grad = torch.stack([torch.full((64, 128), n, device="cuda", dtype=torch.float32) for n in (2, 0, 3)])
    for step in range(2):
        weight.grad_added_to_main_grad = False
        layer(inputs, splits).sum().backward()
        assert len(hook_calls) == step + 1
        expected = grad if overwrite else 1 + (step + 1) * grad
        torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)


_EP_EXPERTS, _EP_HIDDEN, _EP_FFN = 8, 128, 128


def _ep_scalar_loop(x, config):
    return torch.stack([fused_nvfp4_qdq(t, compute_nvfp4_amax(t), config) for t in x])


@contextlib.contextmanager
def _ep_prequantized_te_reference(model):
    """Independent STE oracle: quantize TE leaf storage, run TE, restore full precision.

    No Miles adapter or custom autograd is involved in this reference. TE writes
    directly into its original parameter's main_grad and signals MCore DDP.
    """
    originals = []
    with torch.no_grad():
        for name in ("linear_fc1", "linear_fc2"):
            weight = getattr(model.module.experts, name).weight
            storage = weight.rowwise_data
            originals.append((storage, storage.clone()))
            quantized = _ep_scalar_loop(storage.view(weight.shape), current_nvfp4_qdq_config())
            storage.copy_(quantized.reshape_as(storage))
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "0")
            yield
    finally:
        with torch.no_grad():
            for storage, original in originals:
                storage.copy_(original)


def _make_ep_layer(packed, fused, ep, pg):
    # Megatron is needed only for explicitly requested distributed validation.
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=_EP_HIDDEN,
        num_attention_heads=4,
        num_moe_experts=_EP_EXPERTS,
        moe_ffn_hidden_size=_EP_FFN,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_load_balancing_type="none",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=packed,
        moe_single_grouped_weight=packed,
        use_transformer_engine_op_fuser=False,
        expert_model_parallel_size=ep,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        gradient_accumulation_fusion=fused,
    )
    spec = get_gpt_layer_with_transformer_engine_submodules(num_experts=_EP_EXPERTS, moe_grouped_gemm=True).mlp
    layer = MoELayer(config, submodules=get_submodules(spec), pg_collection=pg)
    layer = Float16Module(config, layer).module.cuda()
    layer.set_layer_number(0)
    expected_first = dist.get_rank(pg.ep) * (_EP_EXPERTS // ep)
    assert layer.local_expert_indices == list(range(expected_first, expected_first + _EP_EXPERTS // ep))
    with torch.no_grad():
        layer.router.weight.zero_()
        for expert in range(_EP_EXPERTS):
            layer.router.weight[expert, expert] = 1
        for fc_name, shape in [("linear_fc1", (2 * _EP_FFN, _EP_HIDDEN)), ("linear_fc2", (_EP_HIDDEN, _EP_FFN))]:
            fc = getattr(layer.experts, fc_name)
            values = []
            for expert in layer.local_expert_indices:
                generator = torch.Generator().manual_seed(90210 + expert * 17 + shape[0])
                values.append((torch.randn(shape, generator=generator) * (0.025 + expert * 0.002)).cuda().bfloat16())
            if packed:
                fc.weight.rowwise_data.view(len(values), *shape).copy_(torch.stack(values))
                assert not fc.weight.allreduce
            else:
                for i, value in enumerate(values):
                    getattr(fc, f"weight{i}").copy_(value)
    return DDP(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=True, grad_reduce_in_fp32=True),
        layer,
        pg_collection=pg,
    )


def _ep_named_values(layer, grad=False):
    values = {"router": layer.router.weight.main_grad if grad else layer.router.weight}
    for name in ("linear_fc1", "linear_fc2"):
        fc = getattr(layer.experts, name)
        if hasattr(fc, "weight"):
            p = fc.weight
            values[name] = (p.main_grad if grad else p.rowwise_data).reshape(p.shape)
        else:
            ps = [getattr(fc, f"weight{i}") for i in range(layer.num_local_experts)]
            values[name] = torch.stack([p.main_grad if grad else p for p in ps])
    return values


def _assert_ep_close(actual, expected, exact):
    a, b = actual.float(), expected.float()
    if exact:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        error = (a - b).abs()
        rms = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        peak = (error.max() / b.abs().max().clamp_min(1e-12)).item()
        assert rms < 0.015 and peak < 0.02, (rms, peak, error.max().item())


def _ep_update(ddp):
    changed = 0
    with torch.no_grad():
        for p in ddp.module.parameters():
            storage = p.rowwise_data if hasattr(p, "rowwise_data") else p
            before = storage.clone()
            storage.add_(p.main_grad.reshape(storage.shape).to(storage.dtype), alpha=-0.05)
            changed += int(torch.count_nonzero(storage != before))
    return changed


@pytest.mark.usefixtures("grouped_qat_env")
@pytest.mark.parametrize("use_4over6", [False, True], ids=["default", "4over6"])
@pytest.mark.parametrize("fused", [False, True], ids=["unfused-wgrad", "fused-wgrad"])
def test_grouped_megatron_ep_forward_backward_and_update(monkeypatch, grouped_ep, use_4over6, fused):
    ep, pg = grouped_ep
    rank = dist.get_rank()
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights" if use_4over6 else "none")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MSE")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    # Legacy discrete QDQ is the non-fused baseline. The fused baseline bypasses
    # Miles STE and runs TE directly on scalar-QDQ values in the original leaves.
    qdq_calls = 0

    def checked_grouped(x, amax, config):
        nonlocal qdq_calls
        output = fused_grouped_nvfp4_qdq(x, amax, config)
        torch.testing.assert_close(output, _ep_scalar_loop(x, config), rtol=0, atol=0)
        qdq_calls += 1
        return output

    monkeypatch.setattr(qdq_kernels, "fused_grouped_nvfp4_qdq", checked_grouped)
    actual = _make_ep_layer(True, fused, ep, pg)
    reference = _make_ep_layer(fused, fused, ep, pg)
    initial_ids = [id(p) for p in actual.module.parameters()]
    for step in range(3 if fused else 1):
        route = "skew_empty" if step == 1 else "balanced"
        actual.zero_grad_buffer()
        reference.zero_grad_buffer()
        for microbatch in range(2):
            generator = torch.Generator(device="cuda").manual_seed(4000 + rank * 101 + step * 11 + microbatch)
            data = torch.randn(32 + rank * 3, 1, _EP_HIDDEN, device="cuda", generator=generator).bfloat16() * 0.1
            data[..., :_EP_EXPERTS] = -1
            primary = (
                ((torch.arange(data.shape[0], device="cuda") + rank) % _EP_EXPERTS)
                if route == "balanced"
                else torch.zeros(data.shape[0], device="cuda", dtype=torch.long)
            )
            data[torch.arange(data.shape[0]), 0, primary] = 4
            data[torch.arange(data.shape[0]), 0, (primary + 1) % _EP_EXPERTS] = 2
            x, y = data.clone().requires_grad_(), data.clone().requires_grad_()
            outputs = []
            for model, input_ in ((actual, x), (reference, y)):
                sync = model.no_sync() if microbatch == 0 else contextlib.nullcontext()
                oracle = (
                    _ep_prequantized_te_reference(model) if model is reference and fused else contextlib.nullcontext()
                )
                with sync, oracle:
                    out, bias = model(input_)
                    assert bias is None
                    (out.float().square().sum() / 10).backward()
                    outputs.append(out)
            _assert_ep_close(*outputs, exact=fused)
            _assert_ep_close(x.grad, y.grad, exact=fused)
        actual.finish_grad_sync()
        reference.finish_grad_sync()
        for name, grad in _ep_named_values(actual.module, grad=True).items():
            assert torch.isfinite(grad).all()
            _assert_ep_close(grad, _ep_named_values(reference.module, grad=True)[name], exact=fused)
            group = pg.dp if name == "router" else pg.expt_dp
            replicas = [torch.empty_like(grad) for _ in range(dist.get_world_size(group))]
            dist.all_gather(replicas, grad.contiguous(), group=group)
            for replica in replicas:
                torch.testing.assert_close(grad, replica, rtol=0, atol=0)
            if name == "router":
                assert torch.count_nonzero(grad) > 0
            else:
                for local, expert in enumerate(actual.module.local_expert_indices):
                    if route == "skew_empty" and expert >= 2:
                        assert torch.count_nonzero(grad[local]) == 0
                    else:
                        assert torch.count_nonzero(grad[local]) > 0
        if fused:
            for fc in (actual.module.experts.linear_fc1, actual.module.experts.linear_fc2):
                assert fc.weight.grad_added_to_main_grad
        changed = _ep_update(actual)
        _ep_update(reference)
        for name, value in _ep_named_values(actual.module).items():
            _assert_ep_close(value, _ep_named_values(reference.module)[name], exact=fused)
        assert [id(p) for p in actual.module.parameters()] == initial_ids
        # Router replicas update even when a rank has no expert tokens.
        assert changed > 0
        assert qdq_calls == 4 * (step + 1)


@pytest.fixture(scope="module", params=[2, 4], ids=["ep2-edp2", "ep4"])
def grouped_ep(request):
    # Requires the Miles hook and Megatron-LM#6000's native GroupedTensor path.
    # Run: torchrun --standalone --nproc_per_node=4 -m pytest <this file> -k grouped_megatron_ep
    if int(os.getenv("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun --nproc_per_node=4")

    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        parallel_state.initialize_model_parallel(expert_model_parallel_size=request.param)
        model_parallel_cuda_manual_seed(123)
        yield request.param, ProcessGroupCollection.use_mpu_process_groups()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
