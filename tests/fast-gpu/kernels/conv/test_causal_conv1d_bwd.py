"""The sequence-aligned conv backward against a float64 torch reference: packed sequences of the lengths its
tiling has edge cases for (one token, shorter than the conv window, not a multiple of the 64-row tile, empty), a
channel count that leaves the last 64-channel strip partly empty, an empty batch, a strided q/k/v slice, fp32
and bf16 weights, and bit-for-bit repeatability. dx is bf16, so it matches to bf16 rounding; dweight is
accumulated in fp32."""

import random
import sys

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("fla")

from miles.kernels.conv.causal_conv1d_bwd import TILE_ROWS, WIDTH, causal_conv1d_bwd, is_available  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not is_available(), reason="causal_conv1d_bwd_cuda extension not built"),
]

CHANNELS = 520
_rng = random.Random(0)


def _lengths(total, choices):
    out = []
    while sum(out) < total:
        out.append(min(_rng.choice(choices), total - sum(out)))
    return out


LENGTHS = {
    "one sequence": [8192],
    "one sequence, unaligned": [8000],
    "even sequences": [512] * 16,
    "unaligned lengths": _lengths(8000, [37, 129, 300, 1001, 2049]),
    "lengths 1 to 5": _lengths(4096, [1, 2, 3, 4, 5, 64, 200]),
    "boundary in every tile": _lengths(8192, [17, 33, 61, 90]),
    "empty sequences": [0, 1, 2, 0, 700, 0, 300],
}


def _inputs(lengths, weight_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)
    x = torch.randn(1, tokens, CHANNELS, device="cuda", dtype=torch.bfloat16, generator=g)
    dout = torch.randn(1, tokens, CHANNELS, device="cuda", dtype=torch.bfloat16, generator=g)
    weight = (torch.randn(CHANNELS, WIDTH, device="cuda", generator=g) * 0.5).to(weight_dtype)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return x, weight, dout, cu_seqlens


def _chunk_indices(cu_seqlens):
    from fla.ops.utils import prepare_chunk_indices

    return prepare_chunk_indices(cu_seqlens, TILE_ROWS)


def _reference(x, weight, dout, cu_seqlens):
    """float64 per-sequence depthwise conv + SiLU through F.conv1d; returns dx, dweight."""
    xr = x[0].double().requires_grad_()
    wr = weight.double().requires_grad_()
    bounds = cu_seqlens.tolist()
    segments = [
        F.silu(F.conv1d(F.pad(xr[start:end].t()[None], (WIDTH - 1, 0)), wr[:, None], groups=CHANNELS))[0].t()
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
        if end > start
    ]
    torch.cat(segments).backward(dout[0].double())
    return xr.grad[None], wr.grad


def _rel_err(actual, expected):
    return ((actual.double() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16], ids=["fp32-weight", "bf16-weight"])
@pytest.mark.parametrize("name", list(LENGTHS))
def test_matches_float64_reference(name, weight_dtype):
    x, weight, dout, cu_seqlens = _inputs(LENGTHS[name], weight_dtype)
    dx, dweight = causal_conv1d_bwd(x, weight, dout, cu_seqlens, _chunk_indices(cu_seqlens))
    ref_dx, ref_dweight = _reference(x, weight, dout, cu_seqlens)
    assert dx.dtype == torch.bfloat16 and dweight.dtype == weight_dtype
    assert _rel_err(dx, ref_dx) < 5e-3
    assert _rel_err(dweight, ref_dweight) < (1e-4 if weight_dtype == torch.float32 else 5e-3)


def test_empty_batch():
    x, weight, dout, cu_seqlens = _inputs([0], torch.float32)
    dx, dweight = causal_conv1d_bwd(x, weight, dout, cu_seqlens, _chunk_indices(cu_seqlens))
    assert dx.shape == x.shape
    assert torch.equal(dweight, torch.zeros_like(weight))


def test_strided_slice_matches_contiguous():
    """x and dout as the middle third of a fused [T, 3C] q/k/v projection, as the layer hands them over."""
    lengths = LENGTHS["unaligned lengths"]
    _, weight, _, cu_seqlens = _inputs(lengths, torch.float32)
    g = torch.Generator(device="cuda").manual_seed(1)
    fused_x = torch.randn(1, sum(lengths), 3 * CHANNELS, device="cuda", dtype=torch.bfloat16, generator=g)
    fused_dout = torch.randn_like(fused_x)
    x, dout = fused_x[..., CHANNELS : 2 * CHANNELS], fused_dout[..., CHANNELS : 2 * CHANNELS]
    chunk_indices = _chunk_indices(cu_seqlens)
    strided = causal_conv1d_bwd(x, weight, dout, cu_seqlens, chunk_indices)
    contiguous = causal_conv1d_bwd(x.contiguous(), weight, dout.contiguous(), cu_seqlens, chunk_indices)
    assert all(torch.equal(a, b) for a, b in zip(strided, contiguous, strict=True))


def test_repeatable():
    """dweight partials are summed in a fixed order, so repeated calls agree bit for bit."""
    x, weight, dout, cu_seqlens = _inputs(LENGTHS["unaligned lengths"], torch.float32)
    chunk_indices = _chunk_indices(cu_seqlens)
    first = causal_conv1d_bwd(x, weight, dout, cu_seqlens, chunk_indices)
    for _ in range(3):
        again = causal_conv1d_bwd(x, weight, dout, cu_seqlens, chunk_indices)
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
