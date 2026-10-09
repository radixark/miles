"""DeepSeek-V4.1 indexer_select with the canonical top-k against the torch top-k it replaces.

The canonical path skips the clean pass and returns each row's picks in ascending order with -1 at the tail;
the torch path sorts -1 to the front. Both must pick the same key set per query (scores are continuous, so
there are no ties), across query chunks, a batch of 2, fewer keys than topk, and candidate-block masking.
"""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.indexer import indexer_select  # noqa: E402

HEADS, DIM, TOPK = 32, 128, 512


def _select(q, index_k, weights, compress_lens, candidates, topk_backend, *, is_candidate_source, query_chunk):
    canonical = topk_backend == "canonical"
    return indexer_select(
        q,
        index_k,
        weights,
        compress_lens,
        candidates,
        is_candidate_source=is_candidate_source,
        uses_candidates=candidates is not None,
        candidate_topk_blocks=4,
        candidate_block_size=128,
        topk=TOPK,
        topk_fn=get_dsa_topk_fn(topk_backend),
        canonical_topk=canonical,
        allow_deep_select=False,
        query_chunk=query_chunk,
    )


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
