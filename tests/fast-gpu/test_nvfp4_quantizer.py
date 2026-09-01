from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-c-8-gpu-b200",
    labels=["precision", "megatron"],
)


import json
import os
import sys
from types import ModuleType

import pytest
import safetensors
import safetensors.torch
import torch
import transformer_engine.pytorch as te
from tools.convert_hf_to_nvfp4 import convert_nvfp4
from tools.convert_hf_to_nvfp4 import quantize_nvfp4 as tool_quantize_nvfp4
from tools.convert_hf_to_nvfp4 import should_quantize as tool_should_quantize_nvfp4
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef

import miles.utils.nvfp4_fake_qat as nvfp4_qat
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import (
    quantize_nvfp4 as processor_quantize_nvfp4,
)
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    current_nvfp4_qdq_config,
    fake_nvfp4_quantization_ste,
    fused_nvfp4_qdq,
)
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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
