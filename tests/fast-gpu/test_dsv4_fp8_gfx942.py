"""Single-GPU FP8 regression tests; run with HIP_VISIBLE_DEVICES=1 pytest <this file>.

These are operator tests, not DeepSeek-V4 end-to-end training/rollout validation.
The CPU E4M3FN reference deliberately does not use the HIP conversion kernels.
"""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_rocm_ci

register_rocm_ci(est_time=60, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])


@pytest.fixture(autouse=True)
def gfx942_only():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm gfx942")
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("regression targets the gfx942 FN/FNUZ conversion")


def _cpu_reference(x, block_size, scale_fmt):
    blocks = x.detach().cpu().float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scales = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    if scale_fmt is not None:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    y = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = (y.float() * scales.unsqueeze(-1)).reshape_as(x).to(x.dtype)
    return y.reshape_as(x), scales, out


def _input(dtype, noncontiguous):
    torch.manual_seed(2026)
    x = torch.randn(33, 2, 1024, device="cuda", dtype=dtype)
    # Outliers, zero blocks, tiny magnitudes, and ragged row count are all
    # important: FNUZ saturation can hide when normalized values stay <240.
    x[0].zero_()
    x[1].mul_(1e-6)
    x[2].mul_(1000)
    x[3, :, ::16] = 448
    return x[..., ::2] if noncontiguous else x[..., :512].contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_act_quant_matches_cpu_fn_bytes(dtype, block_size, scale_fmt, noncontiguous):
    from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import act_quant

    x = _input(dtype, noncontiguous)
    before = x.clone()
    y, scales = act_quant(x, block_size, scale_fmt)
    ref_y, ref_scales, _ = _cpu_reference(x, block_size, scale_fmt)
    assert y.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(y.cpu().view(torch.uint8), ref_y.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), ref_scales, rtol=1e-7, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("scale_fmt", ["ue8m0", None])
def test_qat_matches_cpu_and_preserves_ste(dtype, block_size, scale_fmt, monkeypatch):
    import miles_plugins.models.deepseek_v4.ops.qat as qat

    def broken_tilelang_path(*args, **kwargs):
        pytest.fail("gfx942 must not call the FNUZ cast-back kernel on genuine FN data")

    monkeypatch.setattr(qat, "per_token_cast_back", broken_tilelang_path)
    x = _input(dtype, noncontiguous=True).detach().requires_grad_(True)
    _, _, ref = _cpu_reference(x, block_size, scale_fmt)
    out = qat.fp8_simulate_qat(x, block_size, scale_fmt)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    if scale_fmt == "ue8m0":
        # Existing callers keep the old two-argument behavior exactly.
        torch.testing.assert_close(out, qat.fp8_simulate_qat(x, block_size), rtol=0, atol=0)
    assert torch.isfinite(out).all()
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "ratio,rotate,mode",
    [(ratio, False, mode) for ratio in (4, 128) for mode in ("legacy", "off", "fp8_ue8m0")]
    + [(4, True, mode) for mode in ("legacy", "off", "fp8_ue8m0", "fp8_dynamic")],
)
def test_compressor_qat_policies_match_cpu_reference(packed, ratio, rotate, mode):
    """Real gfx942 compressor with raw/THD inputs; prepared for an idle GPU, not CPU CI."""
    from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
    from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout

    # Explicit policies must work without TE FP8; legacy still follows fp8.
    config = SimpleNamespace(
        hidden_size=32,
        qk_pos_emb_head_dim=64,
        layernorm_epsilon=1e-6,
        fp8="e4m3" if mode == "legacy" else None,
        dsv4_kv_qat="off" if rotate else mode,
        dsv4_index_qat=mode if rotate else "off",
        csa_compress_rotary_base=160000,
        original_max_position_embeddings=65536,
        rotary_scaling_factor=4,
        beta_fast=32,
        beta_slow=1,
    )
    compressor = DeepSeekV4Compressor(config, 128 if rotate else 512, ratio, rotate).cuda()
    torch.manual_seed(27)
    with torch.no_grad():
        for name, param in compressor.named_parameters():
            if "norm" not in name:
                param.normal_(0, 0.05)
    rows = 2 * ratio
    x = torch.randn(rows, 1, 32, device="cuda", dtype=torch.bfloat16)
    layout = ThdLayout(torch.tensor([0, rows], device="cuda", dtype=torch.int32), 0, rows) if packed else None
    with torch.no_grad():
        actual = compressor(x, layout)
        compressor.use_fp8_qat = False
        unquantized = compressor(x, layout)
    expected = unquantized.cpu().clone()
    if mode != "off":
        width, block = (128, 128) if rotate else (448, 64)
        _, _, reference = _cpu_reference(unquantized[..., :width], block, None if mode == "fp8_dynamic" else "ue8m0")
        expected[..., :width] = reference
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [64, 128])
def test_inplace_really_rounds_to_fp8(block_size):
    from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import act_quant

    x = _input(torch.bfloat16, noncontiguous=True)
    before = x.clone()
    _, _, ref = _cpu_reference(x, block_size, "ue8m0")
    out = act_quant(x, block_size, "ue8m0", inplace=True)
    assert out is x
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    assert not torch.equal(out, before)


def test_fn_range_and_negative_zero():
    from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import act_quant

    values = [-448, -320, -240, -128, -1, -0.0, 0.0, 1, 128, 240, 320, 448]
    x = torch.tensor(values, device="cuda", dtype=torch.float32).repeat(64)[:128].reshape(1, 128)
    y, scales = act_quant(x, 128, "ue8m0")
    assert scales.item() == 1
    assert torch.isfinite(y.float()).all()
    torch.testing.assert_close(y.float(), x, rtol=0, atol=0)
    assert y.view(torch.uint8)[0, 5].item() == 128  # FN -0, but FNUZ NaN.


def test_empty_qat():
    from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat

    x = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = fp8_simulate_qat(x, 64)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad.shape == x.shape


def test_real_image_auto_selects_supported_fp8():
    from scripts.amd.run_deepseek_v4 import _probe_fp8_capabilities, _resolve_fp8_recipe

    caps = _probe_fp8_capabilities()
    assert caps.tensorwise[0], caps.tensorwise[1]
    expected = "blockwise" if caps.blockwise[0] else "tensorwise"
    assert _resolve_fp8_recipe("auto") == expected
    if not caps.blockwise[0]:
        with pytest.raises(RuntimeError, match="No BF16 fallback"):
            _resolve_fp8_recipe("blockwise")


@pytest.mark.parametrize("grouped", [False, True])
def test_tensorwise_linear_fp8_forward_backward(grouped, monkeypatch):
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8CurrentScaling, Format
    from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase

    torch.manual_seed(123)
    kwargs = dict(params_dtype=torch.bfloat16, device="cuda", bias=False)
    model = te.GroupedLinear(4, 4096, 2048, **kwargs) if grouped else te.Linear(4096, 2048, **kwargs)
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(512, 2048, dtype=torch.bfloat16, device="cuda") / 100

    def forward():
        return model(x, [128] * 4) if grouped else model(x)

    ref = forward()
    ref.backward(dy)
    ref_x = x.grad.detach().float().clone()
    ref_w = [p.grad.detach().float().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    x.grad = None

    # Inspect the real TE GEMM boundary, not just the autocast flag: both
    # operands of fprop, dgrad and wgrad must actually be FP8 tensors.
    module = importlib.import_module(
        "transformer_engine.pytorch.module.grouped_linear" if grouped else "transformer_engine.pytorch.module.linear"
    )
    gemm_name = "general_grouped_gemm" if grouped else "general_gemm"
    original = getattr(module, gemm_name)
    calls = []

    def checked_gemm(a, b, *args, **kwargs):
        tensors = [*a, *b] if grouped else [a, b]
        assert all(isinstance(t, QuantizedTensorBase) for t in tensors), [type(t) for t in tensors]
        calls.append(kwargs.get("layout", "TN"))
        return original(a, b, *args, **kwargs)

    monkeypatch.setattr(module, gemm_name, checked_gemm)
    with te.fp8_autocast(enabled=True, fp8_recipe=Float8CurrentScaling(fp8_format=Format.E4M3)):
        out = forward()
    out.backward(dy)
    torch.cuda.synchronize()
    assert len(calls) >= 3, "must exercise forward, input gradient and weight gradient FP8 GEMMs"

    def error(actual, expected):
        actual, expected = actual.detach().float(), expected.detach().float()
        assert torch.isfinite(actual).all()
        relative = float((actual - expected).norm() / expected.norm())
        assert relative < 0.06, relative
        return relative

    metrics = {
        "grouped": grouped,
        "out_rel_l2": error(out, ref),
        "dx_rel_l2": error(x.grad, ref_x),
        "dw_rel_l2": [error(p.grad, reference) for p, reference in zip(model.parameters(), ref_w)],
        "fp8_gemm_calls": len(calls),
    }
    print(json.dumps(metrics), flush=True)
