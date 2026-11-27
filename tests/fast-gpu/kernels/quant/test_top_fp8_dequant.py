"""TOP's fused block-FP8 dequantization (miles_plugins.top.kernels.fp8_dequant) against the eager decode it replaces:
the same BF16 bits for packed UE8M0 and FP32 block scales, partial 128-row and 128-column blocks, stacked experts,
strided scale layouts, and every E4M3 code and scale exponent (0, 255, overflow)."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.top.fp8_forward import _dequantize_eager, dequantize_processed  # noqa: E402


def _operands(lead, n, k, *, exponents=(113, 121), seed=0):
    """Every E4M3 byte (NaN codes included) and UE8M0 exponents drawn from ``exponents`` (inclusive)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    payload = torch.randint(0, 256, (*lead, n, k), generator=g, device="cuda", dtype=torch.uint8)
    blocks = ((n + 127) // 128, (k + 127) // 128)
    exponent = torch.randint(exponents[0], exponents[1] + 1, (*lead, *blocks), generator=g, device="cuda")
    return payload.view(torch.float8_e4m3fn), exponent.to(torch.int32)


def _packed_rows(exponent, n):
    """Serving's expanded form: four exponent bytes per int32 word (group 4w + j in byte j), one row per weight row."""
    padded = torch.nn.functional.pad(exponent, (0, -exponent.shape[-1] % 4)).view(*exponent.shape[:-1], -1, 4)
    words = sum(padded[..., j].to(torch.int64) << (8 * j) for j in range(4))
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.repeat_interleave(128, -2)[..., :n, :].contiguous()


def _fp32_blocks(exponent):
    # 0 and 255 have no FP32 power of two; the FP32 layout carries checkpoint scales, which are finite and normal.
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def _assert_same_bits(payload, scale):
    n, k = payload.shape[-2:]
    expected = _dequantize_eager(payload.reshape(-1, n, k), scale.reshape(-1, *scale.shape[-2:]))
    assert torch.equal(
        dequantize_processed(payload, scale).view(torch.int16), expected.view(payload.shape).view(torch.int16)
    )


SHAPES = [
    ((), 256, 384),
    ((), 320, 640),
    ((), 256, 200),
    ((), 200, 136),
    ((), 64, 128),
    ((3,), 192, 520),
    ((2, 2), 128, 256),
]


@pytest.mark.parametrize("lead,n,k", SHAPES)
@pytest.mark.parametrize("layout", ["packed", "fp32"])
def test_fused_dequant_is_the_eager_decode(lead, n, k, layout):
    payload, exponent = _operands(lead, n, k)
    _assert_same_bits(payload, _packed_rows(exponent, n) if layout == "packed" else _fp32_blocks(exponent))


@pytest.mark.parametrize("lead,n,k", [((), 320, 640), ((4,), 256, 1024)])
def test_fused_dequant_every_scale_exponent(lead, n, k):
    """Exponent 0 (2^-127, subnormal products), 255 (NaN) and large exponents (overflow to inf)."""
    payload, exponent = _operands(lead, n, k, exponents=(0, 255), seed=1)
    _assert_same_bits(payload, _packed_rows(exponent, n))
    _assert_same_bits(payload, _fp32_blocks(exponent))


def test_fused_dequant_reads_strided_scales():
    """DeepGEMM's MN-major scale layouts (e.g. DeepSeek-V4's grouped wo_a) are read through their strides."""
    payload, exponent = _operands((2,), 256, 1280, seed=2)
    packed = _packed_rows(exponent, 256)
    mn_major = packed.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not mn_major.is_contiguous()
    _assert_same_bits(payload, mn_major)
    blocks = _fp32_blocks(exponent).transpose(-1, -2).contiguous().transpose(-1, -2)
    _assert_same_bits(payload, blocks)


def test_fused_dequant_reads_serving_scales():
    """The packed scales as serving's transform_scale_ue8m0 lays them out (prepare_weight / synced_weight)."""
    fp8_utils = pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    payload, exponent = _operands((2,), 384, 1280, seed=4)
    served = fp8_utils.transform_scale_ue8m0(_fp32_blocks(exponent), mn=384)
    assert torch.equal(served, _packed_rows(exponent.clamp(1, 254), 384))
    _assert_same_bits(payload, served)


def test_fused_dequant_gate_shapes():
    """GLM-5.2 dense fc1 and EP4 expert w2 shapes at the gate's scale range."""
    for lead, n, k in (((), 24576, 6144), ((8,), 6144, 2048)):
        payload, exponent = _operands(lead, n, k, exponents=(100, 127), seed=3)
        _assert_same_bits(payload, _packed_rows(exponent, n))
