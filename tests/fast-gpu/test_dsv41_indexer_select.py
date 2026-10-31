"""DeepSeek-V4.1 indexer_select: the canonical top-k picks the same keys per query as the torch top-k,
and a THD-packed stream selected segment by segment picks what each segment picks alone."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=100, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=30, suite="nightly-stage-c-2-gpu-mi350", labels=["precision"])

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.indexer import (  # noqa: E402
    indexer_select,
    indexer_select_segments,
    segment_spans,
)

HEADS, DIM, TOPK = 32, 128, 512
SEGMENTS = (1100, 37, 1, 2401, 75)  # odd lengths; one has no compressed key at ratio 2


def _options(topk_backend, *, is_candidate_source, uses_candidates, query_chunk):
    return dict(
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=4,
        candidate_block_size=128,
        topk=TOPK,
        topk_fn=get_dsa_topk_fn(topk_backend),
        canonical_topk=topk_backend == "canonical",
        allow_deep_select=False,
        query_chunk=query_chunk,
    )


def _select(q, index_k, weights, compress_lens, candidates, topk_backend, *, is_candidate_source, query_chunk):
    options = _options(
        topk_backend,
        is_candidate_source=is_candidate_source,
        uses_candidates=candidates is not None,
        query_chunk=query_chunk,
    )
    return indexer_select(q, index_k, weights, compress_lens, candidates, **options)


@pytest.mark.parametrize("bsz,seqlen,ratio", [(1, 4096, 2), (2, 3001, 1), (1, 1500, 4)])
@pytest.mark.parametrize("query_chunk", [None, 1024])
@pytest.mark.parametrize("candidates", ["none", "source", "user"])
def test_canonical_select_matches_torch_select(bsz, seqlen, ratio, query_chunk, candidates):
    torch.manual_seed(0)
    q = torch.randn(bsz, seqlen, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    n_kv = seqlen // ratio
    index_k = torch.randn(bsz, n_kv, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(bsz, seqlen, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    compress_lens = (torch.arange(seqlen, device="cuda") + 1) // ratio
    candidate_mask = None
    if candidates == "user":
        candidate_mask = torch.rand(bsz, seqlen, n_kv, device="cuda") < 0.3
    options = dict(is_candidate_source=candidates == "source", query_chunk=query_chunk)
    torch_idx, torch_cand = _select(q, index_k, weights, compress_lens, candidate_mask, "torch", **options)
    canonical_idx, canonical_cand = _select(q, index_k, weights, compress_lens, candidate_mask, "canonical", **options)

    assert canonical_idx.dtype == torch_idx.dtype == torch.int64
    assert torch.equal(canonical_idx.sort(dim=-1).values, torch_idx)
    valid = canonical_idx >= 0
    assert torch.equal(valid.int(), valid.int().sort(dim=-1, descending=True).values)
    assert bool((canonical_idx < compress_lens.unsqueeze(-1)).all())
    if candidates == "source":
        assert torch.equal(canonical_cand, torch_cand)


@pytest.mark.parametrize("topk_backend", ["canonical", "torch"])
@pytest.mark.parametrize("candidates", ["none", "source", "user"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_select_segments_matches_each_segment_alone(topk_backend, candidates, ratio):
    torch.manual_seed(0)
    total = sum(SEGMENTS)
    cu = [0] + torch.tensor(SEGMENTS).cumsum(0).tolist()
    thd = ThdLayout(
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        seq_lens=SEGMENTS,
        global_start=0,
        max_seqlen=max(SEGMENTS),
    )
    spans = segment_spans(thd, ratio=ratio, seqlen_local=total)
    q = torch.randn(1, total, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(1, spans[-1].key_end, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, total, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    compress_lens = (torch.cat([torch.arange(n, device="cuda") for n in SEGMENTS]) + 1) // ratio
    width = max(span.key_end - span.key_start for span in spans)
    user_mask = torch.rand(1, total, width, device="cuda") < 0.3 if candidates == "user" else None
    options = _options(
        topk_backend,
        is_candidate_source=candidates == "source",
        uses_candidates=user_mask is not None,
        query_chunk=1024,
    )
    idx, cand = indexer_select_segments(q, index_k, weights, compress_lens, user_mask, spans, **options)

    assert idx.shape == (1, total, TOPK)
    for span in spans:
        rows, n = slice(span.row_start, span.row_end), span.key_end - span.key_start
        if n == 0:
            assert bool((idx[:, rows] == -1).all())
            continue
        alone_idx, alone_cand = indexer_select(
            q[:, rows],
            index_k[:, span.key_start : span.key_end],
            weights[:, rows],
            compress_lens[rows],
            user_mask[:, rows, :n] if user_mask is not None else None,
            **options,
        )
        assert torch.equal(idx[:, rows], torch.where(alone_idx >= 0, alone_idx + span.key_start, -1))
        if candidates == "source":
            assert torch.equal(cand[:, rows, :n], alone_cand)
            assert not cand[:, rows, n:].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
