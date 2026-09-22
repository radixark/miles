import pytest
import torch

tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa import build_pooled_keys, kpool_select_topk  # noqa: E402
from miles.kernels.attention.dsa.kpool import _select_expand_tail, pool_boundaries  # noqa: E402

KPOOL = 4
LENS = [37, 13, 22, 64]


def _cu(lens):
    return torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)


def _pooled_keys_ref(index_k, gate, ape, lens, kpool):
    pooled = torch.zeros(sum(lens) // kpool, index_k.shape[1], device="cuda")
    row, start = 0, 0
    for length in lens:
        for pool in range(length // kpool):
            tokens = slice(start + pool * kpool, start + (pool + 1) * kpool)
            weight = torch.softmax(gate[tokens].float() + ape, dim=0)
            pooled[row] = (weight * index_k[tokens].float()).sum(0)
            row += 1
        start += length
    return pooled


def _token_layout(lens, kpool):
    cu, pool_cu = _cu(lens), pool_boundaries(_cu(lens), kpool)
    token_ids = torch.arange(sum(lens), device="cuda")
    seq = torch.searchsorted(cu, token_ids, right=True) - 1
    base = cu[seq].to(torch.int32)
    return base, pool_cu[seq].to(torch.int32), (token_ids - base).to(torch.int32), int(pool_cu[-1])


def _expand_ref(pool_logits, base, pool_base, local, topk, kpool, out_width):
    group_topk = min(topk // kpool, pool_logits.shape[1])
    scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
    out = torch.full((pool_logits.shape[0], out_width), -1, dtype=torch.int32, device="cuda")
    for t in range(pool_logits.shape[0]):
        b, pb, p = int(base[t]), int(pool_base[t]), int(local[t])
        if p + 1 <= topk:
            out[t, : p + 1] = b + torch.arange(p + 1)
            continue
        for g in range(group_topk):
            if torch.isfinite(scores[t, g]):
                out[t, g * kpool : (g + 1) * kpool] = b + (int(pools[t, g]) - pb) * kpool + torch.arange(kpool)
        tail = (p + 1) % kpool
        out[t, topk : topk + tail] = b + (p + 1) // kpool * kpool + torch.arange(tail)
    return out


def test_pooled_keys_match_torch():
    torch.manual_seed(0)
    total, dim = sum(LENS), 128
    index_k = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(KPOOL, dim, device="cuda")
    pooled = build_pooled_keys(index_k, gate, ape, _cu(LENS), KPOOL)
    ref = _pooled_keys_ref(index_k, gate, ape, LENS, KPOOL)
    num_pools = ref.shape[0]
    torch.testing.assert_close(pooled[:num_pools].float(), ref, rtol=1e-2, atol=1e-2)
    assert not pooled[num_pools:].any()


def test_select_expands_pools_and_appends_the_unfinished_pool():
    torch.manual_seed(0)
    topk = 16
    base, pool_base, local, num_pools = _token_layout(LENS, KPOOL)
    logits = torch.randn(sum(LENS), num_pools, device="cuda")
    pool_ids = torch.arange(num_pools, device="cuda")
    eligible = (pool_ids >= pool_base[:, None]) & (pool_ids < (pool_base + (local + 1) // KPOOL)[:, None])
    logits = logits.masked_fill(~eligible, float("-inf"))
    out = _select_expand_tail(logits, base, pool_base, local, topk, KPOOL)
    assert torch.equal(out, _expand_ref(logits, base, pool_base, local, topk, KPOOL, out.shape[1]))


def test_replay_wrapper_path_selects_the_same_tokens():
    torch.manual_seed(0)
    total, heads, dim, topk = sum(LENS), 8, 128, 16
    cu = _cu(LENS)
    index_q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(total, heads, device="cuda") * 0.01
    pooled = build_pooled_keys(index_k, gate, torch.zeros(KPOOL, dim, device="cuda"), cu, KPOOL)
    direct = kpool_select_topk(index_q, pooled, weights, cu, topk, KPOOL)
    wrapped = kpool_select_topk(index_q, pooled, weights, cu, topk, KPOOL, wrap_topk=lambda fn: fn)
    assert direct.shape == (total, 1, direct.shape[-1])
    assert torch.equal(direct, wrapped)
