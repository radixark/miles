"""Chunked DSV4 indexer fwd+topk is bit-exact against the unchunked path.

The unchunked path pre-allocates ``[batch, seqlen_global, seqlen_kv]`` fp32 --
35.25 GiB at S=194560, 64.0 GiB at S=262144 -- which is the ctx128k indexer
OOM. Above a byte threshold ``V4Indexer.forward`` switches to
``batched_indexer_topk_chunked``, whose peak is
``O(q_chunk * seqlen_kv)``.

The equivalence argument is that top-k is row-independent: output row ``p``
depends only on that row's logits, which depend only on ``q[p]``,
``weights[p]``, ``cu_seqlen_ks[p]``, ``cu_seqlen_ke[p]`` and all of ``k``.
Chunking the query dimension therefore cannot change any row. Reductions run
over ``dim``/``heads``, never over the query axis, so this is claimed as
BIT-exact rather than merely close -- and asserted that way here, on real
kernels on a real GPU. A CPU mock cannot substitute: the whole risk is that
the tilelang kernel behaves differently on a sliced/contiguous-ified query
tensor than on the full one.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cuda_ci

# One GPU is enough; stage-b-2-gpu-h200 is the always-run GPU bucket its
# fast-gpu siblings use (tests/ci/run_suite.py), so this lands in CI rather
# than in a suite name nothing schedules.
register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=[])

import pytest
import torch

try:
    import tilelang  # noqa: F401
except ImportError:
    tilelang = None

if tilelang is not None:
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd import (
        _make_causal_cu_seqlens,
        batched_indexer_fwd,
        batched_indexer_topk_chunked,
    )
    from miles_plugins.models.dsa_topk import get_dsa_topk_fn
else:
    _make_causal_cu_seqlens = batched_indexer_fwd = batched_indexer_topk_chunked = get_dsa_topk_fn = None


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
requires_tilelang = pytest.mark.skipif(tilelang is None, reason="tilelang not installed")


def _make_inputs(seqlen, batch, heads, dim, compress_ratio, seed=0):
    torch.manual_seed(seed)
    seqlen_kv = seqlen // compress_ratio
    q = torch.randn(seqlen, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seqlen, batch, heads, device="cuda", dtype=torch.float32) * 0.01
    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen, seqlen_kv, compress_ratio, q.device)
    return q, k, weights, cu_ks, cu_ke, seqlen_kv


def _unchunked(q, k, weights, cu_ks, cu_ke, topk_fn, topk_count):
    index_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)
    return topk_fn(index_scores, topk_count)


# (seqlen, batch, heads, dim, compress_ratio, topk, q_chunk)
_CONFIGS = [
    # q_chunk divides seqlen exactly
    (512, 1, 8, 128, 4, 64, 128),
    (512, 2, 8, 128, 4, 64, 256),
    # q_chunk does NOT divide seqlen -- a ragged last chunk is the boundary case
    (512, 1, 8, 128, 4, 64, 200),
    (384, 2, 16, 128, 4, 32, 100),
    # one chunk covering everything: must degenerate to the unchunked answer
    (256, 1, 8, 128, 4, 32, 256),
    # chunk larger than seqlen
    (256, 1, 8, 128, 4, 32, 4096),
    # q_chunk == 1: the pathological extreme
    (64, 1, 8, 128, 4, 8, 1),
    # C128 layer type: tiny KV, so topk exceeds the valid range for most rows
    # and the -1 padding path is the one under test
    (1024, 1, 16, 128, 128, 8, 256),
    (256, 1, 8, 128, 128, 2, 64),
]
_IDS = [f"sq{s}_b{b}_h{h}_cr{cr}_top{tk}_qc{qc}" for s, b, h, _, cr, tk, qc in _CONFIGS]


@requires_gpu
@requires_tilelang
@pytest.mark.parametrize("seqlen,batch,heads,dim,compress_ratio,topk,q_chunk", _CONFIGS, ids=_IDS)
def test_chunked_topk_is_bit_exact(seqlen, batch, heads, dim, compress_ratio, topk, q_chunk):
    q, k, weights, cu_ks, cu_ke, seqlen_kv = _make_inputs(seqlen, batch, heads, dim, compress_ratio)
    topk_fn = get_dsa_topk_fn("torch")
    topk_count = min(topk, seqlen_kv)

    expected = _unchunked(q, k, weights, cu_ks, cu_ke, topk_fn, topk_count)
    got = batched_indexer_topk_chunked(
        q, k, weights, cu_ks, cu_ke, topk_fn=topk_fn, topk_count=topk_count, q_chunk=q_chunk
    )

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype == torch.int32
    assert torch.equal(got, expected), (
        "chunked indexer top-k diverged from the unchunked path: "
        f"{(got != expected).sum().item()} of {expected.numel()} indices differ"
    )


@requires_gpu
@requires_tilelang
def test_row_order_is_preserved_across_chunk_boundaries():
    """A transposed or off-by-one chunk write would still produce a
    plausible-looking tensor; compare row by row so the failure names the row."""
    seqlen, batch, heads, dim, compress_ratio, topk, q_chunk = 512, 2, 8, 128, 4, 64, 150
    q, k, weights, cu_ks, cu_ke, seqlen_kv = _make_inputs(seqlen, batch, heads, dim, compress_ratio, seed=7)
    topk_fn = get_dsa_topk_fn("torch")
    topk_count = min(topk, seqlen_kv)

    expected = _unchunked(q, k, weights, cu_ks, cu_ke, topk_fn, topk_count)
    got = batched_indexer_topk_chunked(
        q, k, weights, cu_ks, cu_ke, topk_fn=topk_fn, topk_count=topk_count, q_chunk=q_chunk
    )

    for b in range(batch):
        for p in range(seqlen):
            assert torch.equal(got[b, p], expected[b, p]), f"row differs at batch={b}, query position={p}"


@requires_gpu
@requires_tilelang
def test_shape_and_topk_count_are_exactly_as_requested():
    seqlen, batch, heads, dim, compress_ratio = 512, 3, 8, 128, 4
    q, k, weights, cu_ks, cu_ke, seqlen_kv = _make_inputs(seqlen, batch, heads, dim, compress_ratio, seed=3)
    topk_count = 37  # deliberately not a power of two and not a chunk divisor

    got = batched_indexer_topk_chunked(
        q, k, weights, cu_ks, cu_ke, topk_fn=get_dsa_topk_fn("torch"), topk_count=topk_count, q_chunk=128
    )

    assert got.shape == (batch, seqlen, topk_count)
    assert got.dtype == torch.int32
    assert got.device == q.device


@requires_gpu
@requires_tilelang
def test_causally_masked_rows_are_padded_with_minus_one():
    """Early query positions have fewer valid compressed groups than ``topk``.
    Both paths mask those slots to -1; a chunked path that lost the mask would
    emit arbitrary in-range indices and silently widen attention."""
    seqlen, batch, heads, dim, compress_ratio, topk = 256, 1, 8, 128, 4, 64
    q, k, weights, cu_ks, cu_ke, seqlen_kv = _make_inputs(seqlen, batch, heads, dim, compress_ratio, seed=11)
    topk_fn = get_dsa_topk_fn("torch")

    got = batched_indexer_topk_chunked(
        q, k, weights, cu_ks, cu_ke, topk_fn=topk_fn, topk_count=topk, q_chunk=64
    )

    # query position p sees compressed groups [0, (p+1)//compress_ratio) -- so
    # p=0 sees NONE and its whole row must be -1.
    for p in (0, 1, 10, 63):
        expected_valid = min((p + 1) // compress_ratio, topk)
        row = got[0, p]
        assert int((row != -1).sum()) == expected_valid, f"row {p}: expected {expected_valid} valid slots"
        if expected_valid:
            assert int(row[:expected_valid].min()) >= 0, f"row {p}: valid slots must come first"
        else:
            assert int((row == -1).sum()) == topk, f"row {p}: a fully masked row must be all -1"


@requires_gpu
@requires_tilelang
def test_peak_memory_stays_below_the_unchunked_allocation():
    """The reason the chunked path exists. Measured, not asserted structurally:
    a chunked loop that still allocated the full tensor somewhere (e.g. a
    reused buffer sized to seqlen) would pass every equivalence test above.
    """
    seqlen, batch, heads, dim, compress_ratio, topk, q_chunk = 4096, 1, 8, 128, 4, 128, 256
    q, k, weights, cu_ks, cu_ke, seqlen_kv = _make_inputs(seqlen, batch, heads, dim, compress_ratio, seed=5)
    topk_fn = get_dsa_topk_fn("torch")
    full_logits_bytes = batch * seqlen * seqlen_kv * 4

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    batched_indexer_topk_chunked(
        q, k, weights, cu_ks, cu_ke, topk_fn=topk_fn, topk_count=min(topk, seqlen_kv), q_chunk=q_chunk
    )
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated() - base

    assert chunked_peak < full_logits_bytes, (
        f"chunked peak {chunked_peak} B is not below the full logits tensor {full_logits_bytes} B -- "
        "something is still materializing the quadratic score matrix"
    )
