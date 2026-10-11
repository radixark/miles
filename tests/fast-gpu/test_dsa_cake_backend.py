"""Generated deterministic DSA kernels (``--dsa-attention-backend cake``) vs the TileLang kernels and an FP32 reference.

Covers both model contracts on Blackwell (SM100a / SM103a):

* GLM-5 / DeepSeek-V3.2 ``thd`` MLA (``d_qk = 512 + 64``, no sink): ``sparse_attention`` vs ``SparseMLA`` and
  ``lightning_indexer`` vs ``lighting_indexer`` (logits, selected scores, gradients).
* DeepSeek-V4 ``bshd`` MQA (``d = 512``, FP32 sink, batched): ``sparse_attention`` vs ``sparse_attn_tilelang`` and
  the batched ``sbhd`` indexer logits vs ``batched_indexer_fwd``.
* The generated FP32 row top-k (``--miles-dsa-topk-backend cake``, ``-k topk``): exact match with a CPU reference
  of its contract, bit-identical repeated launches, agreement with ``torch.topk`` / FlashInfer up to boundary ties,
  on ``thd``-style flattened rows and through the DeepSeek-V4.1 ``indexer_select`` seam.

Every cake gradient is checked to be bit-identical across two passes.  The TileLang comparisons skip when
``tilelang`` is not importable; the FP32-reference and determinism checks always run.
"""

import math

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-c-4-gpu-b200", labels=["megatron"], hardware=["blackwell"])

try:
    import tilelang  # noqa: F401

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

LOG2E = 1.4426950408889634


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the generated DSA kernels target SM100a / SM103a")


def _cake():
    _require_blackwell()
    from miles.kernels.attention.dsa import cake

    return cake


def _rel_err(actual, expected):
    expected = expected.float()
    actual = actual.float()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), "finite/-inf pattern differs"
    scale = expected[finite].abs().max().clamp_min(1e-6)
    return ((actual[finite] - expected[finite]).abs().max() / scale).item()


def _causal_indices(rows, topk, num_keys, device, generator, offset=0):
    """Row ``r`` selects ``min(topk, r + 1)`` distinct keys ``<= r`` (scaled to ``num_keys``); ``-1`` pads."""
    idx = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    for r in range(rows):
        limit = max(1, min(num_keys, (r + 1) * num_keys // rows))
        count = min(topk, limit)
        perm = torch.randperm(limit, generator=generator, device=device)[:count]
        idx[r, :count] = perm.int() + offset
    return idx


# --------------------------------------------------------------------------- #
# FP32 references (Megatron / cuDNN semantics: -inf sink disables, empty rows -> 0)
# --------------------------------------------------------------------------- #


def ref_sparse_attention(q, kv, indices, sm_scale, attn_sink=None, d_v=512):
    """Flat contract: q [T, H, d_qk], kv [T_kv, d_qk], indices [T, topk] (global, -1 padded)."""
    q32, kv32 = q.float(), kv.float()
    valid = indices >= 0
    gathered = kv32[indices.clamp(min=0).long()]
    scores = torch.einsum("thd,tkd->thk", q32, gathered) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float().view(1, -1, 1).expand(scores.shape[0], -1, 1)
        logits = torch.cat([scores, sink], dim=-1)
    else:
        logits = scores
    row_max = logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp = torch.exp(logits - row_max)
    probs = exp / exp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.einsum("thk,tkd->thd", probs[..., : scores.shape[-1]], gathered[..., :d_v])


def ref_indexer_logits(index_q, index_k, weights, ks, ke):
    sims = torch.relu(torch.einsum("thd,kd->thk", index_q.float(), index_k.float()))
    logits = torch.einsum("thk,th->tk", sims, weights.float())
    keys = torch.arange(index_k.shape[0], device=index_k.device)
    inside = (keys[None, :] >= ks[:, None]) & (keys[None, :] < ke[:, None])
    return logits.masked_fill(~inside, float("-inf"))


def ref_indexer_scores(index_q, index_k, weights, topk_indices):
    valid = topk_indices >= 0
    gathered = index_k.float()[topk_indices.clamp(min=0).long()]
    sims = torch.relu(torch.einsum("thd,tkd->thk", index_q.float(), gathered))
    return torch.einsum("thk,th->tk", sims, weights.float()).masked_fill(~valid, float("-inf"))


# --------------------------------------------------------------------------- #
# GLM-5 / DeepSeek-V3.2: thd MLA
# --------------------------------------------------------------------------- #


def _glm5_attention_inputs(rows, heads, topk, device):
    gen = torch.Generator(device=device).manual_seed(1)
    d_qk = 576
    q = torch.randn(rows, heads, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(rows, 1, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
    indices = _causal_indices(rows, topk, rows, device, gen).unsqueeze(1)  # [T, 1, topk] as glm5.py passes it
    do = torch.randn(rows, heads, 512, device=device, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, do, 1.0 / math.sqrt(d_qk)


@pytest.mark.parametrize("heads", [16, 64], ids=["tp4-h16", "tp1-h64"])
def test_glm5_sparse_attention_matches_reference_and_is_deterministic(heads):
    cake = _cake()
    q, kv, indices, do, sm_scale = _glm5_attention_inputs(512, heads, 256, "cuda")

    def run():
        q_ = q.clone().requires_grad_(True)
        kv_ = kv.clone().requires_grad_(True)
        out = cake.sparse_attention(q_, kv_, indices, sm_scale=sm_scale, layout="thd")
        out.backward(do)
        return out.detach(), q_.grad, kv_.grad

    out, dq, dkv = run()
    out2, dq2, dkv2 = run()
    assert torch.equal(out, out2) and torch.equal(dq, dq2) and torch.equal(dkv, dkv2), "cake pass is not bit-identical"

    q32 = q.clone().float().requires_grad_(True)
    kv32 = kv.squeeze(1).clone().float().requires_grad_(True)
    ref = ref_sparse_attention(q32, kv32, indices.squeeze(1), sm_scale)
    ref.backward(do.float())
    assert _rel_err(out, ref) < 1e-2
    assert _rel_err(dq, q32.grad) < 1e-2
    assert _rel_err(dkv.squeeze(1), kv32.grad) < 1e-2

    if HAS_TILELANG:
        from miles.kernels.attention.dsa import sparse_attention

        q_t = q.clone().requires_grad_(True)
        kv_t = kv.clone().requires_grad_(True)
        out_t = sparse_attention(q_t.unsqueeze(0), kv_t.unsqueeze(0), indices.unsqueeze(0), sm_scale, d_v=512).squeeze(
            0
        )
        out_t.backward(do)
        assert _rel_err(out, out_t) < 1e-2
        assert _rel_err(dq, q_t.grad) < 1e-2
        assert _rel_err(dkv, kv_t.grad) < 1e-2


def _glm5_indexer_inputs(rows, heads, topk, device):
    gen = torch.Generator(device=device).manual_seed(2)
    index_q = torch.randn(rows, heads, 128, device=device, dtype=torch.bfloat16, generator=gen)
    index_k = torch.randn(rows, 128, device=device, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(rows, heads, device=device, generator=gen)
    ks = torch.zeros(rows, device=device, dtype=torch.int32)
    ke = (torch.arange(rows, device=device) + 1).int()
    return index_q, index_k, weights, ks, ke


@pytest.mark.parametrize("heads", [32, 64])
def test_glm5_lightning_indexer_matches_reference_and_is_deterministic(heads):
    cake = _cake()
    rows, topk = 512, 128
    index_q, index_k, weights, ks, ke = _glm5_indexer_inputs(rows, heads, topk, "cuda")

    logits = cake.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    assert _rel_err(logits, ref_indexer_logits(index_q, index_k, weights, ks, ke)) < 1e-3

    def run():
        q_ = index_q.clone().requires_grad_(True)
        k_ = index_k.clone().requires_grad_(True)
        w_ = weights.clone().requires_grad_(True)
        scores, topk_indices = cake.lightning_indexer(q_, k_, w_, ks, ke, topk, layout="thd", topk_backend="torch")
        grad = torch.randn_like(scores).masked_fill(~torch.isfinite(scores), 0.0)
        torch.manual_seed(3)
        grad = torch.randn_like(scores).masked_fill(~torch.isfinite(scores), 0.0)
        scores.masked_fill(~torch.isfinite(scores), 0.0).backward(grad)
        return scores.detach(), topk_indices, q_.grad, k_.grad, w_.grad, grad

    scores, topk_indices, dq, dk, dw, grad = run()
    scores2, topk_indices2, dq2, dk2, dw2, _ = run()
    assert torch.equal(topk_indices, topk_indices2)
    assert torch.equal(scores, scores2) and torch.equal(dq, dq2) and torch.equal(dk, dk2) and torch.equal(dw, dw2)

    q32 = index_q.clone().float().requires_grad_(True)
    k32 = index_k.clone().float().requires_grad_(True)
    w32 = weights.clone().requires_grad_(True)
    ref = ref_indexer_scores(q32, k32, w32, topk_indices)
    assert _rel_err(scores, ref) < 1e-3
    ref.masked_fill(~torch.isfinite(ref), 0.0).backward(grad)
    assert _rel_err(dq, q32.grad) < 2e-2  # ReLU mask flips at |q.k| ~ bf16 eps
    assert _rel_err(dk, k32.grad) < 1e-2
    assert _rel_err(dw, w32.grad) < 1e-3

    if HAS_TILELANG:
        from miles.kernels.attention.dsa import lighting_indexer

        q_t = index_q.clone().requires_grad_(True)
        k_t = index_k.clone().requires_grad_(True)
        w_t = weights.clone().requires_grad_(True)
        scores_t, topk_t = lighting_indexer(q_t, k_t, w_t, ks, ke, topk, topk_fn=lambda logits, k: topk_indices)
        assert torch.equal(topk_t, topk_indices)
        assert _rel_err(scores, scores_t) < 1e-3
        scores_t.masked_fill(~torch.isfinite(scores_t), 0.0).backward(grad)
        assert _rel_err(dq, q_t.grad) < 2e-2
        assert _rel_err(dk, k_t.grad) < 1e-2
        assert _rel_err(dw, w_t.grad) < 1e-3


# --------------------------------------------------------------------------- #
# DeepSeek-V4: bshd MQA with FP32 sink, batched sbhd indexer
# --------------------------------------------------------------------------- #


def _dsv4_attention_inputs(batch, seq, heads, topk, device):
    gen = torch.Generator(device=device).manual_seed(4)
    d = 512
    q = torch.randn(batch, seq, heads, d, device=device, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq, d, device=device, dtype=torch.bfloat16, generator=gen)
    indices = torch.stack([_causal_indices(seq, topk, seq, device, gen) for _ in range(batch)])
    sink = torch.randn(heads, device=device, generator=gen)
    do = torch.randn(batch, seq, heads, d, device=device, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, sink, do, 1.0 / math.sqrt(d)


@pytest.mark.parametrize("heads", [8, 16, 32, 64], ids=["tp8-h8", "tp4-h16", "tp2-h32", "tp1-h64"])
def test_dsv4_sparse_attention_with_sink_matches_reference_and_is_deterministic(heads):
    cake = _cake()
    batch, seq, topk = 2, 256, 192  # 128 window + 64 compressed slots; padded to 256 inside
    q, kv, indices, sink, do, sm_scale = _dsv4_attention_inputs(batch, seq, heads, topk, "cuda")

    def run():
        q_ = q.clone().requires_grad_(True)
        kv_ = kv.clone().requires_grad_(True)
        s_ = sink.clone().requires_grad_(True)
        out = cake.sparse_attention(q_, kv_, indices, sm_scale=sm_scale, attn_sink=s_, layout="bshd")
        out.backward(do)
        return out.detach(), q_.grad, kv_.grad, s_.grad

    out, dq, dkv, dsink = run()
    out2, dq2, dkv2, dsink2 = run()
    assert all(torch.equal(a, b) for a, b in ((out, out2), (dq, dq2), (dkv, dkv2), (dsink, dsink2)))

    flat_idx = torch.where(
        indices >= 0, indices + (torch.arange(batch, device="cuda") * seq).view(batch, 1, 1), indices
    )
    q32 = q.reshape(batch * seq, heads, -1).float().requires_grad_(True)
    kv32 = kv.reshape(batch * seq, -1).float().requires_grad_(True)
    s32 = sink.clone().requires_grad_(True)
    ref = ref_sparse_attention(q32, kv32, flat_idx.reshape(batch * seq, topk), sm_scale, s32)
    ref.backward(do.reshape(batch * seq, heads, -1).float())
    assert _rel_err(out.reshape(batch * seq, heads, -1), ref) < 1e-2
    assert _rel_err(dq.reshape(batch * seq, heads, -1), q32.grad) < 1e-2
    assert _rel_err(dkv.reshape(batch * seq, -1), kv32.grad) < 1e-2
    assert _rel_err(dsink, s32.grad) < 1e-2

    if HAS_TILELANG:
        from miles.kernels.attention.dsa import sparse_attention

        q_t = q.clone().requires_grad_(True)
        kv_t = kv.clone().requires_grad_(True)
        s_t = sink.clone().requires_grad_(True)
        out_t = sparse_attention(q_t, kv_t.unsqueeze(2), indices.unsqueeze(2), sm_scale, attn_sink=s_t)
        out_t.backward(do)
        # Measured on GB300 with TileLang 0.1.9: o 2.4e-3, dq 2.1e-3, dkv 4.2e-3, dsink 5.0e-4 (BF16-level).
        for name, ours, theirs in (
            ("o", out, out_t),
            ("dq", dq, q_t.grad),
            ("dkv", dkv, kv_t.grad),
            ("dsink", dsink, s_t.grad),
        ):
            err = _rel_err(ours, theirs)
            print(f"dsv4 attention {name}: cake vs tilelang rel err {err:.3e}")
            assert err < 1e-2, name


def test_dsv4_sparse_attention_output_allows_inplace_rope_before_backward():
    """DeepSeek-V4 applies the inverse RoPE in place on the attention output before the output projection."""
    cake = _cake()
    q, kv, indices, sink, do, sm_scale = _dsv4_attention_inputs(2, 128, 16, 64, "cuda")
    q_ = q.clone().requires_grad_(True)
    out = cake.sparse_attention(q_, kv, indices, sm_scale=sm_scale, attn_sink=sink, layout="bshd")
    assert out.shape == q.shape and out._base is None, "the output must be a fresh layout-shaped tensor, not a view"
    out[..., -64:].mul_(-1.0)  # in-place on a slice, as apply_rotary_emb(o[..., -rd:], ..., inverse=True) does
    out.backward(do)
    assert q_.grad is not None and torch.isfinite(q_.grad).all()
    # The saved forward output must not see the in-place update: same gradient as the out-of-place form.
    q_ref = q.clone().requires_grad_(True)
    out_ref = cake.sparse_attention(q_ref, kv, indices, sm_scale=sm_scale, attn_sink=sink, layout="bshd")
    scale = torch.ones_like(out_ref)
    scale[..., -64:] = -1.0
    (out_ref * scale).backward(do)
    assert torch.equal(q_.grad, q_ref.grad)


def test_dsv4_batched_indexer_logits_match_reference_and_tilelang():
    cake = _cake()
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(5)
    seq, batch, heads, ratio = 256, 2, 64, 4
    seq_kv = seq // ratio
    q = torch.randn(seq, batch, heads, 128, device=device, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq_kv, batch, 128, device=device, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(seq, batch, heads, device=device, generator=gen)
    positions = torch.arange(seq, device=device, dtype=torch.int32)
    ks = torch.zeros(seq, device=device, dtype=torch.int32)
    ke = ((positions + 1) // ratio).to(torch.int32)

    logits = cake.indexer_logits(q, k, weights, ks, ke, layout="sbhd")
    assert logits.shape == (batch, seq, seq_kv)
    assert torch.equal(logits, cake.indexer_logits(q, k, weights, ks, ke, layout="sbhd"))
    for b in range(batch):
        assert _rel_err(logits[b], ref_indexer_logits(q[:, b], k[:, b], weights[:, b], ks, ke)) < 1e-3

    if HAS_TILELANG:
        from miles.kernels.attention.dsa import indexer_logits_sbhd

        theirs = indexer_logits_sbhd(q, k, weights.float(), ks, ke)
        assert _rel_err(logits, theirs) < 1e-3


# --------------------------------------------------------------------------- #
# Deterministic FP32 row top-k (--miles-dsa-topk-backend cake)
# --------------------------------------------------------------------------- #


def ref_topk(logits, topk):
    """CPU reference of the cake top-k contract in float64 / int64.

    Slot ``j`` holds the key with the ``j``-th highest score (``torch.sort(stable=True)`` descending, so equal
    scores keep ascending index order), ``-inf`` and NaN are never selected and the unfilled tail is ``-1``.
    """
    x = logits.detach().to(device="cpu", dtype=torch.float64)
    selectable = ~(torch.isnan(x) | (x == -math.inf))
    keys = torch.where(selectable, x, torch.full_like(x, -math.inf))
    order = torch.sort(keys, dim=-1, descending=True, stable=True).indices
    take = min(topk, x.shape[1])
    picked = torch.gather(selectable, 1, order)[:, :take]
    out = torch.where(picked, order[:, :take], torch.full_like(order[:, :take], -1))
    return torch.nn.functional.pad(out, (0, topk - take), value=-1).to(torch.int32)


def _topk_logits(rows, num_keys, seed):
    """FP32 rows with exact ties (every third row quantized to half-integers), random valid prefix lengths like
    causal / compressed-length masks (row 0 all ``-inf``, row 1 fewer finite keys than any tested top-k, row 2
    full) and 5 % scattered ``-inf`` inside the prefix."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, num_keys, generator=gen)
    tie_rows = torch.arange(rows) % 3 == 0
    x[tie_rows] = torch.round(x[tie_rows] * 2) / 2
    lens = torch.randint(0, num_keys + 1, (rows,), generator=gen)
    lens[0], lens[1], lens[2] = 0, min(num_keys, 7), num_keys
    x = x.masked_fill(torch.arange(num_keys)[None, :] >= lens[:, None], -math.inf)
    x = x.masked_fill(torch.rand(rows, num_keys, generator=gen) < 0.05, -math.inf)
    return x.cuda()


def _picks(logits, indices):
    """Per row: ``(sorted selected scores, selected keys strictly above the k-th score)``.

    De-tie rule: backends must agree on the multiset of selected scores, but keys tied with the k-th (lowest
    selected) score may be chosen arbitrarily by ``torch.topk`` / FlashInfer while the cake kernel takes the
    smallest indices, so those keys are excluded from the index-set comparison.
    """
    x = logits.detach().to(device="cpu", dtype=torch.float64)
    idx = indices.detach().to(device="cpu", dtype=torch.int64)
    out = []
    for r in range(x.shape[0]):
        picks = idx[r][idx[r] >= 0]
        assert picks.unique().numel() == picks.numel(), f"row {r} selects a key twice"
        scores = x[r, picks]
        above = picks[scores > scores.min()] if picks.numel() else picks
        out.append((scores.sort().values.tolist(), set(above.tolist())))
    return out


_TOPK_CASES = [
    # (rows, num_keys, topk): thd-style flattened rows as lightning_indexer / select_topk feed the backend
    *((rows, num_keys, topk) for rows in (37, 256) for num_keys in (512, 513, 4096, 8191) for topk in (64, 512)),
    (37, 100, 512),  # topk > num_keys: the host pads -inf columns, the kernel writes the -1 tail
    (37, 12000, 512),  # capacity-16384 variant
    (37, 16385, 64),  # capacity-32768 variant
    (37, 32768, 512),  # the maximum row length
]


@pytest.mark.parametrize(("rows", "num_keys", "topk"), _TOPK_CASES, ids=lambda value: str(value))
def test_cake_topk_matches_reference_and_is_bit_identical(rows, num_keys, topk):
    cake = _cake()
    from miles.kernels.attention.dsa.topk import get_dsa_topk_fn, torch_dsa_topk

    logits = _topk_logits(rows, num_keys, seed=rows * 100003 + num_keys * 7 + topk)
    expected = ref_topk(logits, topk)
    out = cake.cake_topk(logits, topk)
    assert out.dtype == torch.int32 and out.shape == (rows, topk)
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(get_dsa_topk_fn("cake")(logits, topk), out)
    assert torch.equal(cake.select_topk(logits, topk, backend="cake"), out)
    for _ in range(19):
        assert torch.equal(cake.cake_topk(logits, topk), out), "cake top-k is not bit-identical across launches"
    theirs = torch_dsa_topk(logits, min(topk, num_keys))
    assert _picks(logits, out) == _picks(logits, theirs)


@pytest.mark.parametrize(("rows", "num_keys", "topk"), [(256, 4096, 512), (37, 8192, 64)])
def test_cake_topk_agrees_with_flashinfer_up_to_ties(rows, num_keys, topk):
    cake = _cake()
    pytest.importorskip("flashinfer")
    logits = _topk_logits(rows, num_keys, seed=11)
    assert _picks(logits, cake.cake_topk(logits, topk)) == _picks(logits, cake.flashinfer_topk(logits, topk))


def test_glm5_lightning_indexer_with_cake_topk_backend():
    """The thd seam: ``lightning_indexer(topk_backend="cake")`` selects per the contract and replay callables pass."""
    cake = _cake()
    rows, heads, topk = 512, 32, 128
    index_q, index_k, weights, ks, ke = _glm5_indexer_inputs(rows, heads, topk, "cuda")
    logits = cake.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    scores, topk_indices = cake.lightning_indexer(
        index_q, index_k, weights, ks, ke, topk, layout="thd", topk_backend="cake"
    )
    assert torch.equal(topk_indices.cpu(), ref_topk(logits, topk))
    assert torch.equal(scores, cake.extract_topk_scores(logits, topk_indices))
    # causal rows: row r sees r + 1 keys, so early rows end in -1 and late rows fill every slot
    assert (topk_indices[0, 1:] == -1).all() and (topk_indices[-1] >= 0).all()
    pinned = topk_indices.flip(-1)
    _, replayed = cake.lightning_indexer(
        index_q, index_k, weights, ks, ke, topk, layout="thd", topk_backend=lambda logits, k: pinned
    )
    assert torch.equal(replayed, pinned)


def _ref_logits_sbhd(q, k, weights, cu_ks, cu_ke):
    """``indexer_logits_sbhd`` in plain torch: ``q [S, B, H, D]``, ``k [S_kv, B, D]``, ``weights [S, B, H]`` ->
    FP32 ``[B, S, S_kv]`` with ``-inf`` outside each query's ``[ks, ke)``."""
    sims = torch.relu(torch.einsum("sbhd,kbd->bshk", q.float(), k.float()))
    logits = torch.einsum("bshk,sbh->bsk", sims, weights.float())
    keys = torch.arange(k.shape[0], device=k.device)
    inside = (keys[None, :] >= cu_ks[:, None]) & (keys[None, :] < cu_ke[:, None])
    return logits.masked_fill(~inside.unsqueeze(0), float("-inf"))


@pytest.mark.parametrize("ratio", [1, 2])
def test_dsv41_indexer_select_with_cake_topk_backend(ratio, monkeypatch):
    """The DeepSeek-V4.1 ``bshd`` seam: compressed-length masking, ``n_kv < topk`` padding, int64 ``-1`` contract."""
    cake = _cake()
    try:
        from miles_plugins.models.deepseek_v4_1.ops import indexer as v41
    except ImportError as exc:  # Megatron / TransformerEngine are not part of every image
        pytest.skip(f"DeepSeek-V4.1 plugin unavailable: {exc}")
    from miles.kernels.attention.dsa.topk import torch_dsa_topk

    # The seam under test is the selection: the logits come from the torch reference (no TileLang dependency).
    monkeypatch.setattr(v41, "indexer_logits_sbhd", _ref_logits_sbhd)
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(6)
    batch, seq, heads, dim, topk = 2, 256, 64, 128, 512
    n_kv = seq // ratio
    q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16, generator=gen)
    index_k = torch.randn(batch, n_kv, dim, device=device, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(batch, seq, heads, device=device, dtype=torch.bfloat16, generator=gen)
    lens = (torch.arange(seq, device=device) + 1) // ratio
    common = dict(
        compress_lens=lens,
        candidates=None,
        is_candidate_source=False,
        uses_candidates=False,
        candidate_topk_blocks=0,
        candidate_block_size=1,
        topk=topk,
        allow_deep_select=False,
        query_chunk=None,
    )
    fed = {}

    def cake_fn(scores, k):
        fed["cake"] = scores
        return cake.cake_topk(scores, k)

    def torch_fn(scores, k):
        fed["torch"] = scores
        return torch_dsa_topk(scores, k)

    idx, _ = v41.indexer_select(q, index_k, weights, topk_fn=cake_fn, topk_backend="cake", **common)
    assert idx.dtype == torch.int64 and idx.shape == (batch, seq, topk)
    assert fed["cake"].shape == (batch * seq, topk) and fed["cake"].dtype == torch.float32  # -inf padded to topk
    expected = ref_topk(fed["cake"], topk).reshape(batch, seq, topk).to(torch.int64)
    assert torch.equal(idx.cpu(), expected), "cake indices are used as returned (no re-sort)"
    assert ((idx < lens.view(1, seq, 1)) | (idx == -1)).all()
    assert torch.equal((idx >= 0).sum(-1), lens.clamp(max=topk).expand(batch, seq))
    idx2, _ = v41.indexer_select(q, index_k, weights, topk_fn=cake_fn, topk_backend="cake", **common)
    assert torch.equal(idx, idx2)

    # The torch path keeps its ascending-index layout and selects the same keys up to boundary ties.
    idx_torch, _ = v41.indexer_select(q, index_k, weights, topk_fn=torch_fn, topk_backend="torch", **common)
    assert torch.equal(fed["torch"], fed["cake"])
    assert torch.equal(idx_torch, idx_torch.sort(dim=-1).values)
    flat_scores = fed["cake"]
    assert _picks(flat_scores, idx.reshape(batch * seq, topk)) == _picks(
        flat_scores, idx_torch.reshape(batch * seq, topk)
    )
