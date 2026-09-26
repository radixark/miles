import importlib.util
import pathlib

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


@pytest.mark.parametrize("cu_seqlens", [[0, 128], [0, 100, 356], [0, 64, 96, 1024]])
@pytest.mark.parametrize("heads", [8, 64])
def test_indexer_logits_packed_matches_reference(cu_seqlens, heads):
    torch.manual_seed(0)
    q, k, weights, ks, ke = _packed_inputs(cu_seqlens, heads, 128)
    ref = reference.indexer_logits_ref(q, k, weights, ks, ke)
    out = indexer_logits(q, k, weights, ks, ke)
    assert torch.equal(torch.isinf(out), torch.isinf(ref))
    finite = ~torch.isinf(ref)
    assert reference.rel_diff(out[finite], ref[finite]) < 1e-3


@pytest.mark.parametrize("seqlen,batch,compress_ratio", [(128, 1, 4), (512, 2, 4), (2048, 1, 128)])
def test_indexer_logits_sbhd_matches_reference(seqlen, batch, compress_ratio):
    torch.manual_seed(0)
    heads, dim = 16, 128
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
    assert scores.shape == topk_indices.shape == (512, 64)
    valid = topk_indices != -1
    gathered = torch.gather(logits, -1, topk_indices.clamp(min=0).long())
    assert torch.equal(scores[valid], gathered[valid])
    # early queries have fewer than 64 valid keys, so padding must be -1 with -inf scores
    assert (topk_indices[0] == -1).sum() == 63
    assert torch.all(scores[~valid] == -torch.inf)
