"""The BF16 stochastic-rounding Triton kernel against a NumPy reference: TE's Philox4x32-10 bits
(checked against the Random123 known-answer vectors), their mapping to elements, and the rounding."""

import numpy as np
import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.quant.stochastic_round import stochastic_round_to_bf16  # noqa: E402

_MASK32 = np.uint64(0xFFFFFFFF)

# philox4x32_10 from Random123's kat_vectors: (counter, key) -> output.
_RANDOM123_KATS = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]

# Seeds and offsets spanning both 32-bit words and the top bit.
_SEED_OFFSETS = [(0, 0), (1234, (7 << 32) | 3), ((1 << 64) - 1, (1 << 64) - 1), (0x9E3779B97F4A7C15, 1 << 63)]


def _philox4x32_10(counter, key):
    """TE's curanddx ``single_round`` x 10 in exact uint64 arithmetic."""
    c0, c1, c2, c3 = np.broadcast_arrays(*(np.asarray(word, dtype=np.uint64) for word in counter))
    k0, k1 = key
    for _ in range(10):
        p0 = c0 * np.uint64(0xD2511F53)
        p1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0),
            p1 & _MASK32,
            (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1),
            p0 & _MASK32,
        )
        k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _reference_bits(src: torch.Tensor, *, seed: int, offset: int) -> torch.Tensor:
    """BF16 bit patterns (int16) the kernel must produce for a CUDA FP32 ``src``."""
    numel = src.numel()
    subsequence = np.arange(-(-numel // 8), dtype=np.uint64)
    words = _philox4x32_10(
        (offset & 0xFFFFFFFF, offset >> 32, subsequence & _MASK32, subsequence >> np.uint64(32)),
        (seed & 0xFFFFFFFF, seed >> 32),
    )
    # Element 8 * s + j takes half j % 2 of word j // 2.
    random16 = np.stack(words, axis=1).astype("<u4").view("<u2").reshape(-1)[:numel].astype(np.uint64)
    bits = src.cpu().numpy().view(np.uint32).astype(np.uint64)
    rounded = torch.from_numpy(
        (((bits + random16) >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.int16)
    )
    rn_cast = src.to(torch.bfloat16).view(torch.int16).cpu()
    return torch.where(torch.isnan(src).cpu(), rn_cast, rounded)


def _round(src: torch.Tensor, *, seed: int, offset: int) -> torch.Tensor:
    out = torch.empty(src.shape, dtype=torch.bfloat16, device=src.device)
    return stochastic_round_to_bf16(src, out, seed=seed, offset=offset)


@pytest.mark.parametrize("counter,key,expected", _RANDOM123_KATS)
def test_reference_philox_matches_random123(counter, key, expected):
    """The reference implements the published Philox4x32-10, so the kernel test anchors to it."""
    assert tuple(int(word) for word in _philox4x32_10(counter, key)) == expected


@pytest.mark.parametrize("numel", [1, 7, 8, 9, 4095, 4096, 4097, (1 << 20) + 3])
@pytest.mark.parametrize("seed,offset", _SEED_OFFSETS)
def test_kernel_matches_reference_bit_for_bit(numel, seed, offset):
    """Every element takes its subsequence, word and half, and rounds exactly as the reference."""
    generator = torch.Generator(device="cuda").manual_seed(numel)
    src = torch.randn(numel, device="cuda", generator=generator) * 0.02
    expected = _reference_bits(src, seed=seed, offset=offset)
    assert torch.equal(_round(src, seed=seed, offset=offset).view(torch.int16).cpu(), expected)


def test_special_values():
    """Zeros, infinities, BF16-representable values, subnormals and the largest finite values keep
    the reference behavior; every NaN, including payloads only in the dropped bits, stays NaN."""
    patterns = [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,  # +-0, +-inf
        0x7FC00000,
        0xFFC00000,
        0x7F800001,
        0xFFFFFFFF,
        0x7FFFFFFF,  # NaNs
        0x3F800000,
        0xBF800000,
        0x3F810000,  # BF16-representable
        0x00000001,
        0x0000FFFF,
        0x007FFFFF,
        0x80000001,  # subnormals
        0x3F7FFFFF,
        0xBF7FFFFF,  # carry into the exponent
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F7F0001,  # near the largest finite values
    ]
    src = torch.tensor(np.array(patterns * 64, dtype=np.uint32).view(np.float32), device="cuda")
    for seed, offset in _SEED_OFFSETS:
        out = _round(src, seed=seed, offset=offset)
        assert torch.equal(out.view(torch.int16).cpu(), _reference_bits(src, seed=seed, offset=offset))
        assert torch.equal(torch.isnan(out), torch.isnan(src))
        representable = ((src.view(torch.int32) & 0xFFFF) == 0) & ~torch.isnan(src)
        assert torch.equal(out[representable].float(), src[representable])


def test_rounds_to_a_neighbor_with_probability_of_its_distance():
    """Each result is one of the two BF16 neighbors, and rounds away from zero as often as the
    dropped bits say: the expectation of the rounded value is the FP32 value."""
    draws = 2048
    fractions = np.array([1, 0x100, 0x1000, 0x4000, 0x8000, 0xC000, 0xFFFF], dtype=np.uint32)
    base = np.array([0x3F80, 0xBF80, 0x0034, 0x4B12], dtype=np.uint32)  # 1, -1, a subnormal, 9.6e6
    bits = ((base[:, None] << np.uint32(16)) | fractions[None, :]).reshape(-1)
    src = torch.from_numpy(bits.view(np.float32)).cuda()
    lower = torch.from_numpy((bits >> np.uint32(16)).astype(np.uint16).view(np.int16)).cuda()
    up_count = torch.zeros_like(src, dtype=torch.int64)
    for offset in range(draws):
        out = _round(src, seed=99, offset=offset).view(torch.int16)
        assert bool(((out == lower) | (out == lower + 1)).all())
        up_count += (out == lower + 1).to(torch.int64)
    probability = torch.from_numpy((bits & np.uint32(0xFFFF)).astype(np.float64)).cuda() / 65536
    sigma = torch.sqrt(draws * probability * (1 - probability)).clamp_min(1.0)
    assert bool(((up_count - draws * probability).abs() <= 5 * sigma).all())


def test_same_arguments_same_bits_other_offsets_other_bits():
    src = torch.randn(1 << 16, device="cuda") * 0.02
    first = _round(src, seed=5, offset=17)
    assert torch.equal(_round(src, seed=5, offset=17), first)
    assert not torch.equal(_round(src, seed=5, offset=18), first)
    assert not torch.equal(_round(src, seed=6, offset=17), first)


def test_rejects_unsupported_arguments():
    src = torch.zeros(16, device="cuda")
    with pytest.raises(TypeError):
        stochastic_round_to_bf16(src.half(), torch.empty(16, dtype=torch.bfloat16, device="cuda"), seed=0, offset=0)
    with pytest.raises(ValueError):
        stochastic_round_to_bf16(src, torch.empty(8, dtype=torch.bfloat16, device="cuda"), seed=0, offset=0)
    with pytest.raises(ValueError):
        stochastic_round_to_bf16(src[::2], torch.empty(8, dtype=torch.bfloat16, device="cuda"), seed=0, offset=0)
    with pytest.raises(ValueError):
        stochastic_round_to_bf16(src, torch.empty(16, dtype=torch.bfloat16, device="cuda"), seed=-1, offset=0)
