"""DeepSeek-V4.1 selector parity, candidate tie policies, and token-index replay isolation."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=100, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=30, suite="nightly-stage-c-2-gpu-mi350", labels=["precision"])

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles.utils.replay_base import IndexerReplayManager  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops import indexer as indexer_ops  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.indexer import indexer_select, select_candidate_blocks  # noqa: E402

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


def _flashinfer_topk(monkeypatch, tie_break):
    if torch.version.hip is not None:
        pytest.skip("FlashInfer top-k requires CUDA")
    pytest.importorskip("flashinfer")
    monkeypatch.setenv("SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK", tie_break)
    return get_dsa_topk_fn("flashinfer")


@pytest.mark.parametrize("tie_break", ["small", "large"])
def test_flashinfer_candidate_ties_and_graph_replay(monkeypatch, tie_break):
    topk_fn = _flashinfer_topk(monkeypatch, tie_break)
    # Five blocks, with a partial final block. Empty rows have no forced block;
    # even an all-masked nonempty row must retain its most recent block.
    width, block_size = 549, 128
    lens = torch.tensor([0, 3, 129, 385, width, width], device="cuda")
    positions = torch.arange(width, device="cuda")
    logits = torch.zeros(2, len(lens), width, device="cuda")
    logits[1, :, block_size : 2 * block_size] = 2
    logits.masked_fill_(positions >= lens[:, None], -torch.inf)
    logits[:, -1] = -torch.inf
    expected = torch.zeros_like(logits, dtype=torch.bool)
    tied_blocks = [[], [0], [0, 1], [0, 3], [0, 4], [4]]
    if tie_break == "large":
        tied_blocks[3:5] = [[2, 3], [3, 4]]
    for batch, blocks_by_row in enumerate([tied_blocks, [[], [0], [0, 1], [1, 3], [1, 4], [4]]]):
        for row, blocks in enumerate(blocks_by_row):
            for block in blocks:
                expected[batch, row, block * block_size : (block + 1) * block_size] = True

    def select():
        candidates = select_candidate_blocks(logits, lens[:, None], 2, block_size, topk_fn=topk_fn)
        indices = topk_fn(logits.masked_fill(~candidates, -torch.inf), 32)
        return candidates, indices

    candidates, indices = select()
    assert torch.equal(candidates, expected)
    assert (indices[:, [0, -1]] == -1).all()
    assert (indices[:, 1] >= 0).sum().item() == 6
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            select()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_candidates, captured_indices = select()
    captured_candidates.zero_()
    captured_indices.fill_(-2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_candidates, candidates)
    assert torch.equal(captured_indices.sort(dim=-1).values, indices.sort(dim=-1).values)


@pytest.mark.parametrize("tie_break", ["small", "large"])
def test_flashinfer_candidate_ties_at_v41_topk_limit(monkeypatch, tie_break):
    topk_fn = _flashinfer_topk(monkeypatch, tie_break)
    # HF1017 uses 2048 candidate blocks of 8 keys. The partial 4097th block
    # exercises the graph-safe FlashInfer limit and a non-power-of-two width.
    width, block_size, topk_blocks = 32769, 8, 2048
    lens = torch.tensor([width - 1, width], device="cuda")
    positions = torch.arange(width, device="cuda")
    logits = torch.zeros(1, 2, width, device="cuda").masked_fill(positions >= lens[:, None], -torch.inf)
    blocks = positions // block_size
    last = (lens[:, None] - 1) // block_size
    if tie_break == "small":
        expected = (blocks < topk_blocks - 1) | (blocks == last)
    else:
        expected = (blocks > last - topk_blocks) & (blocks <= last)

    def select():
        return select_candidate_blocks(logits, lens[:, None], topk_blocks, block_size, topk_fn=topk_fn)

    assert torch.equal(select(), expected.unsqueeze(0))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            select()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select()
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, expected.unsqueeze(0))


@pytest.mark.parametrize("tie_break", ["small", "large"])
@pytest.mark.parametrize("query_chunk", [None, 3])
def test_flashinfer_select_ties_source_and_consumer(monkeypatch, tie_break, query_chunk):
    topk_fn = _flashinfer_topk(monkeypatch, tie_break)
    torch.manual_seed(1)
    bsz, seqlen, n_kv, topk, block_size = 2, 8, 549, 64, 128
    q = torch.randn(bsz, seqlen, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(bsz, n_kv, DIM, device="cuda", dtype=torch.bfloat16)
    # Run the real 32-head/128-dim scorer, with exact ties after head weighting.
    weights = torch.zeros(bsz, seqlen, HEADS, device="cuda")
    lengths = [0, 3, 127, 129, 257, 385, 513, n_kv]
    lens = torch.tensor(lengths, device="cuda")
    options = dict(
        candidate_topk_blocks=2,
        candidate_block_size=block_size,
        topk=topk,
        topk_fn=topk_fn,
        canonical_topk=False,
        allow_deep_select=False,
        query_chunk=query_chunk,
    )
    source_indices, candidates = indexer_select(
        q, index_k, weights, lens, None, is_candidate_source=True, uses_candidates=False, **options
    )
    expected_candidates = torch.zeros_like(candidates)
    for row, length in enumerate(lengths):
        if length:
            last = (length - 1) // block_size
            selected = [last] + ([0 if tie_break == "small" else last - 1] if last else [])
            for block in selected:
                expected_candidates[:, row, block * block_size : (block + 1) * block_size] = True
    assert torch.equal(candidates, expected_candidates)
    # Different masks per batch ensure chunking preserves the batch/query axes.
    candidates[1, :, ::2] = False
    candidates[1, -1] = False
    consumer_indices, _ = indexer_select(
        q, index_k, weights, lens, candidates, is_candidate_source=False, uses_candidates=True, **options
    )
    for actual, mask in [(source_indices, None), (consumer_indices, candidates.cpu())]:
        expected = torch.full(actual.shape, -1, dtype=torch.int64)
        for batch in range(bsz):
            for row, length in enumerate(lengths):
                eligible = [key for key in range(length) if mask is None or mask[batch, row, key]]
                selected = eligible[:topk] if tie_break == "small" else eligible[-topk:]
                expected[batch, row, : len(selected)] = torch.tensor(selected, dtype=torch.int64)
        assert actual.dtype == torch.int64
        assert torch.equal(actual.cpu(), expected.sort(dim=-1).values)


@pytest.mark.parametrize(
    "backend,tie_break", [("torch", None), ("canonical", None), ("flashinfer", "small"), ("flashinfer", "large")]
)
def test_candidate_selection_bypasses_token_replay(monkeypatch, backend, tie_break):
    topk_fn = _flashinfer_topk(monkeypatch, tie_break) if backend == "flashinfer" else get_dsa_topk_fn(backend)
    manager = IndexerReplayManager()
    monkeypatch.setattr(indexer_ops, "indexer_replay_manager", manager)
    manager.enabled = True
    manager.stage = "record"
    replay = manager.create_replay()
    manager.set_current(replay)
    bsz, seqlen, n_kv, topk = 2, 8, 549, 64
    q = torch.zeros(bsz, seqlen, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.zeros(bsz, n_kv, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.zeros(bsz, seqlen, HEADS, device="cuda")
    lens = torch.tensor([0, 3, 127, 129, 257, 385, 513, n_kv], device="cuda")
    score_chunks, final_topk_calls, replay_checks = [], [], []
    logits_fn = indexer_ops.indexer_logits_sbhd
    check_indices = manager.check_replay_indices

    def chunked_logits(q, *args, **kwargs):
        score_chunks.append(q.shape[0])
        assert q.shape[0] <= 3
        return logits_fn(q, *args, **kwargs)

    def counted_topk(scores, *args):
        if scores.ndim == 2:
            final_topk_calls.append(scores.shape[0])
        return topk_fn(scores, *args)

    def check_whole_forward(fresh, recorded):
        assert fresh.shape == recorded.shape == (bsz * seqlen, topk)
        replay_checks.append(1)
        check_indices(fresh, recorded)

    monkeypatch.setattr(indexer_ops, "indexer_logits_sbhd", chunked_logits)
    monkeypatch.setattr(manager, "check_replay_indices", check_whole_forward)

    def select():
        score_chunks.clear()
        final_topk_calls.clear()
        result = indexer_select(
            q,
            index_k,
            weights,
            lens,
            None,
            is_candidate_source=True,
            uses_candidates=False,
            candidate_topk_blocks=2,
            candidate_block_size=128,
            topk=topk,
            topk_fn=counted_topk,
            canonical_topk=backend == "canonical",
            allow_deep_select=False,
            query_chunk=3,
        )
        assert score_chunks == [3, 3, 2]
        return result

    manager.stage = "fallthrough"
    fallthrough_indices, _ = select()
    assert not replay.top_indices_list
    manager.stage = "record"
    recorded_indices, recorded_candidates = select()
    assert torch.equal(recorded_indices, fallthrough_indices)
    assert final_topk_calls == [6, 6, 4]
    # Rollout supplies one whole-forward tensor. Neither candidates nor query chunks add entries.
    assert len(replay.top_indices_list) == 1
    assert replay.top_indices_list[0].shape == (bsz * seqlen, topk)
    assert replay.top_indices_list[0].dtype == torch.int32
    assert (recorded_indices[:, 0] == -1).all()
    # One mismatch passes the whole-forward 10% threshold, but fails a per-chunk check.
    original = replay.top_indices_list[0].clone()
    missing = next(key for key in range(n_kv) if key not in original[-1].tolist())
    replay.top_indices_list[0][-1] = missing
    manager.enable_check_replay_result = True
    monkeypatch.setenv("MILES_TEST_R3_THRESHOLD", "0.1")
    manager.stage = "replay_forward"
    checked_indices, _ = select()
    assert (checked_indices[-1, -1] == missing).all()
    assert replay_checks == [1]
    assert final_topk_calls == [6, 6, 4]
    replay.top_indices_list[0] = original
    replay.clear_forward()
    manager.enable_check_replay_result = False
    if tie_break is not None:
        monkeypatch.setenv("SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK", "large" if tie_break == "small" else "small")
    for stage in ("replay_forward", "replay_backward"):
        manager.stage = stage
        replayed_indices, candidates = select()
        assert torch.equal(replayed_indices, recorded_indices)
        if tie_break is not None:
            # The raw backend still recomputes candidate ties; only final keys come from replay.
            assert not torch.equal(candidates, recorded_candidates)
        assert len(replay.top_indices_list) == 1
        assert not final_topk_calls
    assert replay.forward_index == replay.backward_index == 1
    replay.staged_forward_ids = original.to(q.device)
    manager.stage = "replay_forward"
    for _ in range(2):
        staged_indices, _ = select()
        assert torch.equal(staged_indices, recorded_indices)
        assert not final_topk_calls
    assert replay.forward_index == replay.backward_index == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
