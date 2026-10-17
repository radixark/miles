"""PyTorch arithmetic matching SGLang's Step5 indexer rounding and reduction order.

`tests/manual/step5/indexer_parity.py` compares these stages with the CUDA kernels.
`score_regions` documents the FP32 versus tensor-core accumulation exception.
"""

import math
from dataclasses import dataclass

import torch

from miles_plugins.models.step5.region_selection import REGION_SIZE, TOP_REGIONS

INDEXER_HEAD_DIM = 256
INDEXER_ROPE_DIM = 32
# The scoring kernels hard-code four heads per provider group, and their head sum below.
HEADS_PER_GROUP = 4
# Scores are ``sum_h 0.5 * w_h * relu(q_h . k)``: SGLang's weight scale is heads_per_group ** -0.5.
WEIGHT_SCALE = HEADS_PER_GROUP**-0.5

_ROPE_HALF = INDEXER_ROPE_DIM // 2
_LOG2_E = 1.4426950408889634
_E4M3_MAX = 448.0
_FLT_MIN = torch.finfo(torch.float32).tiny
# Tie ranks ride in the low bits of the selection key, so a sequence may hold up to 2**20 regions.
_REGION_ID_BITS = 20
# SGLang's deep_select/cuda_kernels/common_parts.cuh defines this segment traversal.
_DEEPSELECT_SEGMENT_SIZE = 512
_DEEPSELECT_INIT_WINDOW = 8192
_DEEPSELECT_TAIL_SEGMENTS = 8
_DEEPSELECT_PERM_ADD_BASE = 0x22262226
_DEEPSELECT_PERM_MUL_PRIME = 0xB559EB75


@dataclass(frozen=True)
class Llama3Rope:
    """The indexer's RoPE: the layer's theta with the checkpoint's llama3 ``rope_scaling``."""

    theta: float
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


def rope_cos_sin(rope: Llama3Rope, num_positions: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows ``[0, num_positions)`` of SGLang's BF16 ``cos_sin_cache``, as ``(cos, sin)`` of shape ``[P, 16]``.

    Follows ``Llama3RotaryEmbedding`` op for op in FP32. CUDA and CPU trigonometry differ in the last
    bit, so the table equals SGLang's only when ``device`` is CUDA.
    """
    exponents = torch.arange(0, INDEXER_ROPE_DIM, 2, dtype=torch.float, device=device) / INDEXER_ROPE_DIM
    inv_freq = 1.0 / (rope.theta**exponents)
    low_freq_wavelen = rope.original_max_position_embeddings / rope.low_freq_factor
    high_freq_wavelen = rope.original_max_position_embeddings / rope.high_freq_factor
    wave_len = 2 * math.pi / inv_freq
    smooth = 0
    if rope.low_freq_factor != rope.high_freq_factor:
        smooth = (rope.original_max_position_embeddings / wave_len - rope.low_freq_factor) / (
            rope.high_freq_factor - rope.low_freq_factor
        )
    inv_freq = torch.where(
        wave_len < high_freq_wavelen,
        inv_freq,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freq / rope.factor,
            (1 - smooth) * inv_freq / rope.factor + smooth * inv_freq,
        ),
    )
    freqs = torch.arange(num_positions, dtype=torch.float, device=device)[:, None] * inv_freq
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def norm_rope_query(query: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float):
    """Per-head GemmaRMSNorm ``x * rsqrt(mean(x^2) + eps) * (1 + w)``, then RoPE; BF16 ``[T, heads, 256]``."""
    x = query.float()
    inverse_rms = torch.rsqrt(_triton_sum(x * x) / INDEXER_HEAD_DIM + eps)
    normalized = (x * inverse_rms[..., None]) * (_as_bf16(weight) + 1.0)
    return _rotate(normalized.to(torch.bfloat16), cos[:, None], sin[:, None])


def norm_rope_key(key: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, cos, sin, eps: float):
    """LayerNorm with the kernel's ``E[x^2] - mean^2`` variance, then RoPE; BF16 ``[T, 256]``.

    The affine ``x_hat * w + b`` rounds once: on GB300 the kernel's paired FMUL2/FADD2 give the FMA's
    result, bit for bit, including where ``b`` cancels ``x_hat * w``.
    """
    x = key.float()
    mean = _triton_sum(x) / INDEXER_HEAD_DIM
    variance = _triton_sum(x * x) / INDEXER_HEAD_DIM - mean * mean
    x_hat = (x - mean[..., None]) * torch.rsqrt(variance + eps)[..., None]
    normalized = _fma(x_hat, _as_bf16(weight).expand_as(x_hat), _as_bf16(bias).expand_as(x_hat))
    return _rotate(normalized.to(torch.bfloat16), cos, sin)


def to_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Unscaled E4M3 with round-to-nearest-even, saturating at +-448 (PTX ``cvt.rn.satfinite``)."""
    return values.float().clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)


def compress_regions(key: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """One E4M3 key per complete region: a per-channel softmax over its 8 tokens weighting ``key``.

    ``key`` is the normalized, rotated BF16 index key and ``logits`` the raw BF16 compression logits,
    both ``[regions, 8, 256]``. As in ``csa_ring._compressed_index_key_bytes``: one maximum over the
    whole region, ``ex2.approx.ftz`` weights, an ordered FMA numerator after a rounded first product, a
    zero result for an empty denominator, then FP32 -> BF16 -> E4M3.
    """
    z, value = logits.float(), key.float()
    region_max = z.amax(dim=(1, 2), keepdim=True)
    weight = torch.exp2((z - region_max) * _LOG2_E)
    # ex2.approx.ftz flushes subnormal results; torch.exp2 otherwise matches it exactly.
    weight = torch.where(weight < _FLT_MIN, 0.0, weight)
    numerator = weight[:, 0] * value[:, 0] + 0.0
    denominator = weight[:, 0]
    for token in range(1, REGION_SIZE):
        numerator = _fma(weight[:, token], value[:, token], numerator)
        denominator = denominator + weight[:, token]
    mean = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-20), 0.0)
    return to_e4m3(mean.to(torch.bfloat16))


def score_regions(query: torch.Tensor, weights: torch.Tensor, region_keys: torch.Tensor) -> torch.Tensor:
    """Return weighted ReLU dot scores `[T, groups, regions]` in FP32.

    `query` is E4M3 `[T, groups, 4, 256]`, `region_keys` is E4M3 `[regions, 256]`, and
    FP32 `weights` `[T, groups, 4]` includes `WEIGHT_SCALE`. Head products are rounded
    before the decode kernel's `(h0 + h2) + (h1 + h3)` sum. FP32 dots can differ from
    SGLang tensor-core accumulation in the last bit, changing top-k at score ties.
    """
    dots = torch.matmul(query.float(), region_keys.float().T)
    terms = torch.where(dots > 0, dots, 0.0) * weights[..., None]
    return (terms[:, :, 0] + terms[:, :, 2]) + (terms[:, :, 1] + terms[:, :, 3])


def top_regions(scores: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Ascending ids of the ``TOP_REGIONS`` best of the first ``candidates[t]`` regions of each row.

    ``scores`` is FP32 ``[T, groups, R]`` and every candidate count exceeds ``TOP_REGIONS``. The order is
    DeepSelect's: descending score, NaN first, and ties in DeepSelect's candidate visitation order. That
    visitation is region order through 8,192 candidates; for longer inputs, the tail (up to eight
    512-region segments) comes first, followed by a fixed permutation of the prefix segments. Returns
    int32 ``[T, groups, TOP_REGIONS]``.
    """
    region = torch.arange(scores.shape[-1], device=scores.device)
    tie_rank = _deepselect_visit_rank(candidates, scores.shape[-1])
    ranked = _ordered_keys(scores) * (1 << _REGION_ID_BITS) + ((1 << _REGION_ID_BITS) - 1 - tie_rank[:, None])
    ranked = ranked.masked_fill(region >= candidates[:, None, None], -1)
    return ranked.topk(TOP_REGIONS, dim=-1).indices.sort(dim=-1).values.to(torch.int32)


def _deepselect_visit_rank(candidates: torch.Tensor, width: int) -> torch.Tensor:
    """Invert DeepSelect's tail-first segment traversal without copying candidate counts to the CPU."""
    visit = torch.arange(width, device=candidates.device)[None, :]
    if width <= _DEEPSELECT_INIT_WINDOW:
        return visit

    candidates = candidates.long()[:, None]
    num_segments = (candidates + (_DEEPSELECT_SEGMENT_SIZE - 1)) // _DEEPSELECT_SEGMENT_SIZE
    num_perm_segments = torch.where(
        num_segments <= _DEEPSELECT_INIT_WINDOW // _DEEPSELECT_SEGMENT_SIZE,
        torch.zeros_like(num_segments),
        ((num_segments - 1) // _DEEPSELECT_TAIL_SEGMENTS) * _DEEPSELECT_TAIL_SEGMENTS,
    )
    tail_start = num_perm_segments * _DEEPSELECT_SEGMENT_SIZE
    tail_count = candidates - tail_start
    prefix_visit = (visit - tail_count).clamp_min(0)
    perm_len = num_perm_segments.clamp_min(1)
    segment = (
        (prefix_visit // _DEEPSELECT_SEGMENT_SIZE) * (_DEEPSELECT_PERM_MUL_PRIME % perm_len)
        + _DEEPSELECT_PERM_ADD_BASE
    ) % perm_len
    prefix_region = segment * _DEEPSELECT_SEGMENT_SIZE + prefix_visit % _DEEPSELECT_SEGMENT_SIZE
    visited_region = torch.where(visit < tail_count, tail_start + visit, prefix_region)
    # Keep padding at its own index so the inverse scatter remains one-to-one for every row.
    visited_region = torch.where(visit < candidates, visited_region, visit)
    return torch.empty_like(visited_region).scatter_(1, visited_region, visit.expand_as(visited_region))


def _triton_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last (256) dim in the order of the norm kernel's 4-warp ``tl.sum``.

    Each of the 128 threads adds its two adjacent channels, each warp reduces its 32 lanes by shuffles
    (lane l with lane l + 16, then + 8, ...), and the four warp sums combine as ``(w0 + w2) + (w1 + w3)``.
    """
    lanes = (x[..., 0::2] + x[..., 1::2]).unflatten(-1, (4, 32))
    for width in (16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    warps = lanes[..., 0]
    return (warps[..., 0] + warps[..., 2]) + (warps[..., 1] + warps[..., 3])


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE on channels ``[0, 32)``, pairs ``(c, c + 16)``, in FP32 without FMA; BF16 result."""
    value = x.float()
    first, second = value[..., :_ROPE_HALF], value[..., _ROPE_HALF:INDEXER_ROPE_DIM]
    cos, sin = cos.float(), sin.float()
    rotated = torch.cat([first * cos - second * sin, second * cos + first * sin, value[..., INDEXER_ROPE_DIM:]], -1)
    return rotated.to(torch.bfloat16)


def _as_bf16(weight: torch.Tensor) -> torch.Tensor:
    """SGLang holds every indexer parameter in BF16, rounding the FP32 checkpoint tensors on load."""
    return weight.to(torch.bfloat16).float()


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 ``fma(a, b, c)``: the FP64 product of two FP32 values is exact, so only the sum rounds."""
    return (a.double() * b.double() + c.double()).float()


def _ordered_keys(scores: torch.Tensor) -> torch.Tensor:
    """int64 keys whose order is the scores' numeric order, -0 equal to +0 and NaN above +inf."""
    bits = (scores + 0.0).view(torch.int32).to(torch.int64)
    keys = torch.where(bits >= 0, bits + (1 << 31), -1 - bits)
    return keys.masked_fill(scores.isnan(), (1 << 32) - 1)
