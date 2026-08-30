"""Tensor-core QSA attention: same answer as the gather kernel, orders of magnitude faster.

The gather kernel materialises a ``[BQ, BK, D]`` tile per step and reduces it with ALU
math because each query has its own key set; at the production shape (T=25k, 12 q-heads,
D=256, budget 2048) that measured 2.5 s forward and 13.9 s forward+backward per layer,
which was ~95% of a training micro-batch. The replacement walks key tiles with ``tl.dot``
and masks each (query, key) pair against a per-sequence block bitmap: 23 ms / 118 ms for
the same numbers.

These tests pin the equivalence, including the packed case with a sequence boundary that
is not a multiple of the compress ratio -- the case where a global block grid would mix
one sequence's blocks into another's.
"""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
    pytest.skip("QSA kernels need CUDA", allow_module_level=True)

from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_block_sparse_attn import (  # noqa: E402
    build_tile_index_pair,
    qsa_block_sparse_attention_triton,
    qsa_sparse_attention_from_indices,
    selection_to_block_bitmap,
)
from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_sparse_attn import qsa_sparse_attention_triton  # noqa: E402

BLK = 4
HQ, HKV, D = 4, 1, 128
SCALE = D**-0.5


def _single_sequence_case(T=384, budget=64, seed=0):
    """Selection as the indexer emits it: distinct blocks, tail clamped to the position."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    rows = torch.arange(T, device="cuda")
    nblk = budget // BLK
    scores = torch.rand(T, T // BLK + 1, device="cuda", generator=g)
    allowed = torch.arange(scores.shape[1], device="cuda").unsqueeze(0) <= (rows // BLK).unsqueeze(1)
    scores = torch.where(allowed, scores, torch.full_like(scores, -1.0))
    blk = scores.topk(min(nblk, scores.shape[1]), dim=1).indices
    tok = (blk.unsqueeze(-1) * BLK + torch.arange(BLK, device="cuda")).reshape(T, -1)
    idx = torch.where(tok <= rows.unsqueeze(1), tok, torch.full_like(tok, -1)).to(torch.int32)
    return q, k, v, idx


def _reference(q, k, v, idx, scale):
    """fp64, one query at a time, straight from the documented semantics."""
    T, hq, d = q.shape
    group = hq // k.shape[1]
    out = torch.zeros(T, hq, d, device=q.device, dtype=torch.float64)
    qd, kd, vd = q.double(), k.double(), v.double()
    for t in range(T):
        sel = idx[t][idx[t] >= 0].long()
        if sel.numel() == 0:
            continue
        for h in range(hq):
            s = (qd[t, h].unsqueeze(0) * kd[sel, h // group]).sum(-1) * scale
            p = torch.softmax(s, dim=0)
            out[t, h] = (p.unsqueeze(-1) * vd[sel, h // group]).sum(0)
    return out


def _rel(a, b):
    d = (a.double() - b.double()).abs()
    return d.max().item(), (d.mean() / (b.double().abs().mean() + 1e-12)).item()


def test_matches_gather_kernel_and_reference():
    q, k, v, idx = _single_sequence_case()
    ref = _reference(q, k, v, idx, SCALE)
    old = qsa_sparse_attention_triton(q, k, v, idx, SCALE)
    new = qsa_sparse_attention_from_indices(q, k, v, idx, SCALE)

    old_max, _ = _rel(old, ref)
    new_max, new_rel = _rel(new, ref)
    # neither is closer to exact than the other by more than the bf16 floor
    assert new_max <= old_max * 4, (new_max, old_max)
    assert new_rel < 1e-2, new_rel
    assert _rel(new, old)[1] < 1e-2


def test_gradients_match_gather_kernel():
    q, k, v, idx = _single_sequence_case()
    gout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        fn(qq, kk, vv, idx, SCALE).backward(gout)
        return qq.grad, kk.grad, vv.grad

    for a, b, name in zip(
        run(qsa_sparse_attention_from_indices), run(qsa_sparse_attention_triton), "qkv", strict=True
    ):
        rel = _rel(a, b)[1]
        assert rel < 1e-2, (name, rel)


def test_packed_boundary_not_a_multiple_of_the_block():
    """A sequence starting off the global block grid must not borrow its neighbour's blocks."""
    lens = [301, 211]
    T = sum(lens)
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)

    cu = torch.tensor([0, *lens], device="cuda").cumsum(0)
    tok_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    blk_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    pos = torch.zeros(T, dtype=torch.int64, device="cuda")
    boff = 0
    for i, length in enumerate(lens):
        s0 = int(cu[i])
        tok_base[s0 : s0 + length] = s0
        blk_base[s0 : s0 + length] = boff
        pos[s0 : s0 + length] = torch.arange(length, device="cuda")
        boff += -(-length // BLK)

    sel = torch.zeros(T, boff, dtype=torch.uint8, device="cuda")
    idx = torch.full((T, 64), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        p = int(pos[t])
        cand = torch.arange((p + 1) // BLK, device="cuda")
        chosen = cand[torch.randperm(cand.numel(), device="cuda", generator=g)[:7]] if cand.numel() else cand
        blocks = torch.cat([chosen, torch.tensor([p // BLK], device="cuda")]).unique()
        sel[t, int(blk_base[t]) + blocks] = 1
        toks = (blocks.unsqueeze(1) * BLK + torch.arange(BLK, device="cuda")).reshape(-1) + int(tok_base[t])
        toks = toks[(toks >= int(tok_base[t])) & (toks <= t)]
        idx[t, : toks.numel()] = toks.to(torch.int32)

    out = qsa_block_sparse_attention_triton(
        q, k, v, sel, tok_base, torch.arange(T, dtype=torch.int32, device="cuda"), blk_base, tok_base, SCALE, BLK
    )
    ref = _reference(q, k, v, idx, SCALE)
    assert _rel(out, ref)[1] < 1e-2
    # the second sequence is the one a global grid would corrupt
    assert _rel(out[lens[0] :], ref[lens[0] :])[1] < 1e-2


def test_tile_index_pair_is_consistent_both_ways():
    q, k, v, idx = _single_sequence_case(T=256, budget=32)
    sel = selection_to_block_bitmap(idx, 256, BLK)
    klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, 64, 32, BLK)

    forward = {(qt, int(klist[qt, i])) for qt in range(kcnt.numel()) for i in range(int(kcnt[qt]))}
    backward = {(int(qlist[kt, i]), kt) for kt in range(qcnt.numel()) for i in range(int(qcnt[kt]))}
    assert forward == backward
