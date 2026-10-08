import importlib.util
import pathlib

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa import (  # noqa: E402
    causal_ranges,
    causal_ranges_compressed,
    get_dsa_topk_fn,
    indexer_logits,
    indexer_logits_sbhd,
    indexer_topk_scores,
    lighting_indexer,
)
from miles.kernels.attention.dsa.indexer import IndexerConfig  # noqa: E402
from miles.kernels.attention.dsa.topk import canonical_dsa_topk, select_topk  # noqa: E402

_spec = importlib.util.spec_from_file_location("dsa_reference", pathlib.Path(__file__).with_name("dsa_reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def _packed_inputs(cu_seqlens, heads, dim):
    total = cu_seqlens[-1]
    q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(total, heads, device="cuda", dtype=torch.float32) * 0.01
    ks, ke = causal_ranges(torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32))
    return q, k, weights, ks.int(), ke.int()


_SCORE_CONFIGS = {
    "tilelang": IndexerConfig(score_backend="tilelang"),
    "triton": IndexerConfig(score_backend="triton", block_rows=128, block_n=64, num_warps=4, num_stages=2),
    "triton-wide": IndexerConfig(score_backend="triton", block_rows=256, block_n=128, num_warps=8, num_stages=3),
}


@pytest.mark.parametrize("cu_seqlens", [[0, 128], [0, 100, 356], [0, 64, 96, 1024], [0, 1, 2, 3, 517]])
@pytest.mark.parametrize("heads", [8, 32, 64])
@pytest.mark.parametrize("score_backend", list(_SCORE_CONFIGS))
def test_indexer_logits_packed_matches_reference(cu_seqlens, heads, score_backend):
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs(cu_seqlens, heads, 128)
    ref = reference.indexer_logits_ref(q, k, weights, ks, ke)
    out = indexer_logits(q, k, weights, ks, ke, config=_SCORE_CONFIGS[score_backend])
    assert torch.equal(torch.isinf(out), torch.isinf(ref))
    finite = ~torch.isinf(ref)
    assert reference.rel_diff(out[finite], ref[finite]) < 1e-3
    unclean = indexer_logits(q, k, weights, ks, ke, clean_logits=False, config=_SCORE_CONFIGS[score_backend])
    assert torch.equal(unclean[finite], out[finite])


@pytest.mark.parametrize("seqlen,batch,compress_ratio", [(128, 1, 4), (512, 2, 4), (2048, 1, 128), (1031, 2, 4)])
@pytest.mark.parametrize("heads", [16, 64])
def test_indexer_logits_sbhd_matches_reference(seqlen, batch, compress_ratio, heads):
    torch.manual_seed(0)
    dim = 128
    seqlen_kv = seqlen // compress_ratio
    q = torch.randn(seqlen, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seqlen, batch, heads, device="cuda", dtype=torch.float32) * 0.01
    ks, ke = causal_ranges_compressed(seqlen, compress_ratio, q.device)
    out = indexer_logits_sbhd(q, k, weights, ks, ke)
    for b in range(batch):
        ref = reference.indexer_logits_ref(q[:, b], k[:, b], weights[:, b], ks, ke)
        assert torch.equal(torch.isinf(out[b]), torch.isinf(ref))
        finite = ~torch.isinf(ref)
        assert reference.rel_diff(out[b][finite], ref[finite]) < 1e-3


@pytest.mark.parametrize("topk", [32, 100, 512])
def test_indexer_topk_scores_backward_matches_autograd(topk):
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs([0, 300, 1024], 16, 128)
    logits = indexer_logits(q, k, weights, ks, ke)
    topk_indices = logits.topk(min(topk, logits.shape[-1]), dim=-1).indices.int()
    topk_indices = topk_indices.masked_fill(torch.gather(logits, -1, topk_indices.long()) == -torch.inf, -1)

    q_ref = q.clone().float().requires_grad_()
    k_ref = k.clone().float().requires_grad_()
    w_ref = weights.clone().requires_grad_()
    ref_logits = reference.indexer_logits_ref(q_ref, k_ref, w_ref, ks, ke)
    ref_scores = torch.gather(ref_logits, -1, topk_indices.clamp(min=0).long())
    grad_scores = torch.randn(topk_indices.shape, device="cuda", dtype=torch.float32)
    (torch.where(topk_indices != -1, ref_scores, 0.0) * grad_scores).sum().backward()

    q_tl = q.clone().requires_grad_()
    k_tl = k.clone().requires_grad_()
    w_tl = weights.clone().requires_grad_()
    scores = indexer_topk_scores(q_tl, k_tl, w_tl, logits, topk_indices)
    (torch.where(topk_indices != -1, scores, 0.0) * grad_scores).sum().backward()

    for ref_g, tl_g in ((q_ref.grad, q_tl.grad), (k_ref.grad, k_tl.grad), (w_ref.grad, w_tl.grad)):
        assert reference.rel_diff(ref_g, tl_g) < 1e-4


def test_lighting_indexer_returns_topk_of_its_own_logits():
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs([0, 512], 8, 128)
    scores, topk_indices = lighting_indexer(q, k, weights, ks, ke, topk=64, topk_fn=get_dsa_topk_fn("torch"))
    logits = indexer_logits(q, k, weights, ks, ke)
    canonical_scores, canonical_indices = lighting_indexer(
        q, k, weights, ks, ke, topk=64, topk_fn=get_dsa_topk_fn("canonical"), clean_logits=False
    )
    assert torch.equal(canonical_indices, reference.canonical_topk_ref(logits, 64, ks, ke))
    assert torch.equal(canonical_indices.sort(-1).values, topk_indices.sort(-1).values)
    canonical_valid = canonical_indices != -1
    gathered_canonical = torch.gather(logits, -1, canonical_indices.clamp(min=0).long())
    assert torch.equal(canonical_scores[canonical_valid], gathered_canonical[canonical_valid])
    assert scores.shape == topk_indices.shape == (512, 64)
    valid = topk_indices != -1
    gathered = torch.gather(logits, -1, topk_indices.clamp(min=0).long())
    assert torch.equal(scores[valid], gathered[valid])
    # early queries have fewer than 64 valid keys, so padding must be -1 with -inf scores
    assert (topk_indices[0] == -1).sum() == 63
    assert torch.all(scores[~valid] == -torch.inf)


def _packed_ranges(seq_lens, compress_ratio=1, device="cuda"):
    """Per-row [ks, ke) over packed keys: causal (ratio 1, key t + 1 visible) or compressed (ratio > 1)."""
    starts, ends, key_base = [], [], 0
    for n in seq_lens:
        positions = torch.arange(n, device=device)
        visible = positions + 1 if compress_ratio == 1 else (positions + 1) // compress_ratio
        starts.append(torch.full((n,), key_base, device=device))
        ends.append(key_base + visible)
        key_base += n // compress_ratio if compress_ratio > 1 else n
    return torch.cat(starts).int(), torch.cat(ends).int(), key_base


def _scores(rows, n_kv, ks, ke, tie_step=0.0, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, n_kv, device="cuda", generator=generator)
    if tie_step:
        logits = torch.round(logits / tie_step) * tie_step
    padded = torch.full((rows, -(-n_kv // 4) * 4), float("nan"), device="cuda")
    padded[:, :n_kv] = logits
    columns = torch.arange(n_kv, device="cuda")
    outside = (columns < ks.long().unsqueeze(1)) | (columns >= ke.long().unsqueeze(1))
    # out-of-range columns hold garbage: the top-k must never read them
    padded[:, :n_kv] = torch.where(outside, float("nan"), logits)
    return padded[:, :n_kv], logits


@pytest.mark.parametrize(
    "seq_lens,compress_ratio,topk",
    [
        ([4096], 1, 512),
        ([4096], 4, 512),
        ([3001, 5003, 7919, 461], 1, 2048),
        ([3001, 5003, 7919, 461], 4, 512),
        ([1, 1, 2, 3, 700, 1, 129], 1, 64),
        ([1000], 128, 512),
    ],
    ids=["causal", "compressed", "packed-odd-glm5", "packed-odd-c4", "tiny-segments", "hca-all-short"],
)
@pytest.mark.parametrize("tie_step", [0.0, 0.25], ids=["distinct", "heavy-ties"])
def test_canonical_topk_matches_reference(seq_lens, compress_ratio, topk, tie_step):
    ks, ke, n_kv = _packed_ranges(seq_lens, compress_ratio)
    scores, clean = _scores(ks.shape[0], n_kv, ks, ke, tie_step=tie_step)
    k = min(topk, n_kv)
    got = canonical_dsa_topk(scores, k, ks, ke)
    assert got.dtype == torch.int32 and got.shape == (ks.shape[0], k)
    assert torch.equal(got, reference.canonical_topk_ref(clean, k, ks, ke))
    assert torch.equal(got, canonical_dsa_topk(_scores(ks.shape[0], n_kv, ks, ke, tie_step=tie_step)[0], k, ks, ke))


def test_canonical_topk_set_equals_torch_topk_up_to_ties():
    ks, ke, n_kv = _packed_ranges([3001, 5003, 7919, 461])
    scores, clean = _scores(ks.shape[0], n_kv, ks, ke)
    columns = torch.arange(n_kv, device="cuda")
    in_range = (columns >= ks.long().unsqueeze(1)) & (columns < ke.long().unsqueeze(1))
    cleaned = clean.masked_fill(~in_range, float("-inf"))
    expected = get_dsa_topk_fn("torch")(cleaned, 2048)
    got = canonical_dsa_topk(scores, 2048, ks, ke)
    assert torch.equal(got.sort(-1).values, expected.sort(-1).values)


@pytest.mark.parametrize(
    "proposal_kind",
    ["all-padding", "out-of-range", "duplicates", "one-duplicate", "below-threshold", "shuffled-exact"],
)
@pytest.mark.parametrize("tie_step", [0.0, 0.25], ids=["distinct", "heavy-ties"])
def test_select_topk_is_exact_for_any_proposal(proposal_kind, tie_step):
    """A wrong proposal only costs speed: the in-kernel radix select must still return the exact answer."""
    ks, ke, n_kv = _packed_ranges([1500, 2049, 3, 900])
    scores, clean = _scores(ks.shape[0], n_kv, ks, ke, tie_step=tie_step)
    topk = 256
    expected = reference.canonical_topk_ref(clean, topk, ks, ke)
    generator = torch.Generator(device="cuda").manual_seed(1)
    proposal = {
        "all-padding": lambda: torch.full_like(expected, -1),
        "out-of-range": lambda: torch.where(expected >= 0, expected + 4096, -1),
        "duplicates": lambda: expected[:, :1].expand_as(expected).contiguous(),
        # keeps the k-th value, so only the duplicate check can reject it
        "one-duplicate": lambda: torch.cat([expected[:, :1], expected[:, :1], expected[:, 2:]], dim=1),
        "below-threshold": lambda: torch.where(
            expected >= 0, ke.unsqueeze(1) - 1 - torch.arange(topk, device="cuda"), -1
        )
        .clamp(min=-1)
        .int(),
        "shuffled-exact": lambda: expected[:, torch.randperm(topk, device="cuda", generator=generator)].contiguous(),
    }[proposal_kind]()
    assert torch.equal(select_topk(scores, topk, ks, ke, proposal), expected)


def test_canonical_topk_skips_masked_keys_and_empty_rows():
    """-inf inside a range is a masked key (DeepSeek-V4.1 candidate blocks), never a pick; empty rows are all -1."""
    rows, n_kv, topk = 64, 1024, 128
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = torch.arange(rows, device="cuda", dtype=torch.int32) * 16
    scores, clean = _scores(rows, n_kv, ks, ke)
    masked = torch.rand(rows, n_kv, device="cuda") < 0.9
    scores = scores.masked_fill(masked, float("-inf"))
    clean = clean.masked_fill(masked, float("-inf"))
    got = canonical_dsa_topk(scores, topk, ks, ke)
    assert torch.equal(got, reference.canonical_topk_ref(clean, topk, ks, ke))
    assert torch.all(got[0] == -1)


def test_canonical_topk_beyond_32768_rows():
    """cuDNN's fused indexer top-k returns short rows past row 32768 on H200; this path must stay exact there."""
    rows, n_kv, topk = 40000, 768, 512
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = torch.full((rows,), n_kv, dtype=torch.int32, device="cuda") - torch.arange(rows, device="cuda").int() % 300
    scores, clean = _scores(rows, n_kv, ks, ke)
    got = canonical_dsa_topk(scores, topk, ks, ke)
    assert torch.equal(got, reference.canonical_topk_ref(clean, topk, ks, ke))
    assert torch.equal((got >= 0).sum(-1), (ke - ks).clamp(max=topk))


def test_canonical_topk_rejects_unaligned_rows():
    scores = torch.randn(8, 1027, device="cuda")
    ks = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="SCORE_ROW_ALIGN"):
        canonical_dsa_topk(scores, 16, ks, ks + 1027)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
