"""Generated deterministic DSA kernels (``--miles-dsa-kernel-backend cake``) vs the default DSA kernel dispatch and an
FP32 reference.

Covers both model contracts on Blackwell (SM100a / SM103a):

* GLM-5 / DeepSeek-V3.2 ``thd`` MLA (``d_qk = 512 + 64``, no sink): cake ``sparse_attention`` vs
  ``miles.kernels.attention.dsa.sparse_attention`` and cake ``lightning_indexer`` vs ``lighting_indexer`` (logits,
  selected scores, gradients).
* DeepSeek-V4 ``bshd`` MQA (``d = 512``, FP32 sink, batched): cake ``sparse_attention`` vs the default dispatch and
  the batched ``sbhd`` indexer logits vs ``indexer_logits_sbhd``.
* The opt-in rows of the default dispatch (``CAKE_SPARSE_ATTENTION_CONFIG`` / ``CAKE_INDEXER_CONFIG``, what the model
  plugins pass for ``cake``) return exactly what the cake package returns.

Every cake gradient is checked to be bit-identical across two passes.  The default-dispatch comparisons skip when
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
        scores_t, topk_t = lighting_indexer(
            q_t, k_t, w_t, ks, ke, topk, topk_fn=lambda logits, k, starts, ends: topk_indices
        )
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


@pytest.mark.parametrize("heads", [16, 64], ids=["tp4-h16", "tp1-h64"])
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
# The opt-in rows of the default dispatch (what --miles-dsa-kernel-backend cake selects)
# --------------------------------------------------------------------------- #


def test_dispatch_cake_rows_match_the_cake_package_bitwise():
    """``sparse_attention(config=CAKE_SPARSE_ATTENTION_CONFIG)`` on the [B, S, H, D] / [B, S_kv, 1, D] contract
    equals the cake package call (thd for one sequence, bshd for a batch), output and gradients."""
    cake = _cake()
    from miles.kernels.attention.dsa import sparse_attention
    from miles.kernels.attention.dsa.sparse_attention import CAKE_SPARSE_ATTENTION_CONFIG

    # GLM-5: one packed sequence, RoPE tail, no sink
    q, kv, indices, do, sm_scale = _glm5_attention_inputs(256, 16, 128, "cuda")
    q_a, kv_a = q.clone().requires_grad_(True), kv.clone().requires_grad_(True)
    out_a = sparse_attention(
        q_a.unsqueeze(0),
        kv_a.unsqueeze(0),
        indices.unsqueeze(0),
        sm_scale,
        d_v=512,
        config=CAKE_SPARSE_ATTENTION_CONFIG,
    ).squeeze(0)
    out_a.backward(do)
    q_b, kv_b = q.clone().requires_grad_(True), kv.clone().requires_grad_(True)
    out_b = cake.sparse_attention(q_b, kv_b, indices, sm_scale=sm_scale, layout="thd")
    out_b.backward(do)
    assert torch.equal(out_a, out_b) and torch.equal(q_a.grad, q_b.grad) and torch.equal(kv_a.grad, kv_b.grad)

    # DeepSeek-V4: a batch, FP32 sink
    q, kv, indices, sink, do, sm_scale = _dsv4_attention_inputs(2, 128, 16, 64, "cuda")
    leaves_a = [t.clone().requires_grad_(True) for t in (q, kv, sink)]
    out_a = sparse_attention(
        leaves_a[0],
        leaves_a[1].unsqueeze(2),
        indices.unsqueeze(2),
        sm_scale,
        attn_sink=leaves_a[2],
        config=CAKE_SPARSE_ATTENTION_CONFIG,
    )
    out_a.backward(do)
    leaves_b = [t.clone().requires_grad_(True) for t in (q, kv, sink)]
    out_b = cake.sparse_attention(
        leaves_b[0], leaves_b[1], indices, sm_scale=sm_scale, attn_sink=leaves_b[2], layout="bshd"
    )
    out_b.backward(do)
    assert torch.equal(out_a, out_b)
    for a, b in zip(leaves_a, leaves_b, strict=True):
        assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("seq_len_kv", [512, 509], ids=["aligned", "padded"])
def test_dispatch_cake_indexer_rows_match_the_cake_package_and_feed_the_range_topk(seq_len_kv):
    cake = _cake()
    from miles.kernels.attention.dsa import indexer_logits, indexer_logits_sbhd
    from miles.kernels.attention.dsa.indexer import CAKE_INDEXER_CONFIG, SCORE_ROW_ALIGN
    from miles.kernels.attention.dsa.topk import select_topk

    index_q, index_k, weights, ks, ke = _glm5_indexer_inputs(512, 32, 128, "cuda")
    index_k = index_k[:seq_len_kv].contiguous()
    ke = ke.clamp(max=seq_len_kv)
    logits = indexer_logits(index_q, index_k, weights, ks, ke, config=CAKE_INDEXER_CONFIG)
    assert logits.stride(0) % SCORE_ROW_ALIGN == 0
    assert torch.equal(logits, cake.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd"))
    topk = 64
    proposal = torch.full((logits.shape[0], topk), -1, dtype=torch.int32, device="cuda")
    picked = select_topk(logits, topk, ks, ke, proposal)
    expected = torch.topk(logits, topk, dim=-1)
    expected = torch.where(torch.isfinite(expected.values), expected.indices.int(), -1).sort(dim=-1).values
    assert torch.equal(picked.sort(dim=-1).values, expected)

    # sbhd: one cake launch for the whole batch
    gen = torch.Generator(device="cuda").manual_seed(6)
    seq, batch = 256, 2
    q = torch.randn(seq, batch, 64, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq_len_kv // 4, batch, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(seq, batch, 64, device="cuda", generator=gen)
    positions = torch.arange(seq, device="cuda", dtype=torch.int32)
    ks4 = torch.zeros(seq, device="cuda", dtype=torch.int32)
    ke4 = ((positions + 1) // 4).clamp(max=k.shape[0]).to(torch.int32)
    logits = indexer_logits_sbhd(q, k, w, ks4, ke4, config=CAKE_INDEXER_CONFIG)
    assert logits.stride(1) % SCORE_ROW_ALIGN == 0
    assert torch.equal(logits, cake.indexer_logits(q, k, w, ks4, ke4, layout="sbhd"))
