"""Qwen3.8-Flash-Next triton kernels vs their torch references (1 GPU).

The references are the sglang-parity-verified torch implementations the
production triton kernels replaced: grouped RMSNorm + hyper-connection
mix/inject/combine, the PLE gate+conv chain, sparse attention over explicit
index lists, and the n-gram hash. Everything runs fwd + bwd and compares
against autograd through the reference.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["miles-plugin"])

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.hc_triton import hc_combine_triton, hc_mix_inject_triton
from miles_plugins.models.qwen3_8_next.ops.kernel.ple_triton import ple_gate_conv_triton
from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_sparse_attn import qsa_sparse_attention_triton
from miles_plugins.models.qwen3_8_next.ops.ple import ngram_hash_ids, shift_right_ignore_eos

# ---- torch references ----


def grouped_gemma_rmsnorm(x: Tensor, weight: Tensor, n: int, eps: float) -> Tensor:
    """Per-stream RMSNorm; the scale enters as ``1 + weight``; returns fp32."""
    acc = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xg = x.to(acc).unflatten(-1, (n, x.shape[-1] // n))
    xn = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + eps)).flatten(-2)
    return xn * (1.0 + weight.to(acc))


def hc_mix(normed, w_down, w_up, n, hidden, out_dtype):
    """The ``/ n`` sits before the SiLU; the reduction is a mean over streams."""
    gate = F.silu(F.linear(normed, w_down.to(normed.dtype)) / n)
    gate = torch.sigmoid(F.linear(gate, w_up.to(normed.dtype)))
    mixed = (gate.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(dim=-2)
    return mixed.to(out_dtype)


def hc_inject_gate(normed, w_inject, n):
    return 2 * torch.sigmoid(F.linear(normed, w_inject.to(normed.dtype)) / n)


def hc_combine(residual, block_output, h_post, n, hidden):
    out_dtype = residual.dtype
    R = residual.float().unflatten(-1, (n, hidden))
    injection = block_output.float().unsqueeze(-2) * h_post.float().unsqueeze(-1)
    return (R + injection).flatten(-2).to(out_dtype)


def causal_depthwise_conv(x, weight, dilation, cu_seqlens=None):
    channels, _, kernel = weight.shape
    pad = (kernel - 1) * dilation

    def _conv(seq):
        h = seq.transpose(0, 1).unsqueeze(0)
        h = F.conv1d(F.pad(h, (pad, 0)), weight, groups=channels, dilation=dilation)
        return h.squeeze(0).transpose(0, 1)

    if cu_seqlens is None:
        return _conv(x)
    out = torch.empty_like(x)
    bounds = cu_seqlens.tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=False):
        if hi > lo:
            out[lo:hi] = _conv(x[lo:hi])
    return out


def ple_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu):
    T = hc.shape[0]
    C = hc.shape[1] // n
    kn = grouped_gemma_rmsnorm(key, wk, n, eps).reshape(T, n, C)
    qn = grouped_gemma_rmsnorm(hc, wq, n, eps).reshape(T, n, C)
    score = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(C)
    gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
    gated = (gate * value.unsqueeze(-2)).flatten(-2)
    gn = grouped_gemma_rmsnorm(gated, wc, n, eps)
    conv = F.silu(causal_depthwise_conv(gn.to(convw.dtype), convw, dil, cu))
    return (gated.to(conv.dtype) + conv).to(hc.dtype)


def qsa_reference(q, k, v, indices, scale):
    """Attention over exactly the listed indices (list semantics, unique rows)."""
    T, Hq, D = q.shape
    S, Hkv, _ = k.shape
    rep = Hq // Hkv
    mask = torch.zeros(T, S, dtype=torch.bool, device=q.device)
    valid = indices >= 0
    rows = torch.arange(T, device=q.device).unsqueeze(-1).expand_as(indices)
    mask[rows[valid], indices[valid].long()] = True
    qh = q.transpose(0, 1).float()
    kh = k.transpose(0, 1).repeat_interleave(rep, dim=0).float()
    vh = v.transpose(0, 1).repeat_interleave(rep, dim=0).float()
    scores = torch.einsum("htd,hsd->hts", qh, kh) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), 0.0)
    return torch.einsum("hts,hsd->htd", p, vh).transpose(0, 1)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


# ---- n-gram hash ----

NGRAM_SIZE, HEADS_PER_NGRAM = 3, 8
EOS = 248044
VOCAB = 248320
# The three tensors the checkpoint ships (read out of the safetensors headers).
MULT = [23703573157769, 20109073645365, 8052911324071]
SIZES = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
OFFS = [0]
for s in SIZES[:-1]:
    OFFS.append(OFFS[-1] + s)


def _hash_tensors():
    def t(v):
        return torch.tensor(v, dtype=torch.long, device="cuda")

    return t(MULT), t(SIZES), t(OFFS)


def test_ngram_hash_ids_land_in_each_heads_row_range():
    mult, sizes, offs = _hash_tensors()
    ctx = torch.randint(
        0,
        VOCAB,
        (256, NGRAM_SIZE),
        device="cuda",
        dtype=torch.long,
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    ids = ngram_hash_ids(ctx, mult, sizes, offs, NGRAM_SIZE, HEADS_PER_NGRAM, EOS)
    assert ids.shape == (256, len(SIZES))
    for h in range(ids.shape[-1]):
        lo, hi = OFFS[h], OFFS[h] + SIZES[h]
        col = ids[:, h]
        assert bool(((col >= lo) & (col < hi)).all()), f"head {h} out of [{lo}, {hi})"


@pytest.mark.parametrize("eos_frac", [0.0, 0.25])
def test_ngram_hash_matches_sglang(eos_frac):
    qwen4_exp = pytest.importorskip("sglang.srt.models.qwen4_exp")
    mult, sizes, offs = _hash_tensors()
    g = torch.Generator(device="cuda").manual_seed(1)

    stub = type("Stub", (), {"eos_token_id": EOS})()
    tok = torch.randint(0, VOCAB, (3, 16), device="cuda", dtype=torch.long, generator=g)
    if eos_frac:
        m = torch.rand(3, 16, device="cuda", generator=g) < eos_frac
        tok = torch.where(m, torch.full_like(tok, EOS), tok)
    for n in range(NGRAM_SIZE):
        mine = shift_right_ignore_eos(tok, n, EOS)
        ref = qwen4_exp.Qwen4ExpNGramEmbedding._shift_right_ignore_eos(stub, tok, n)
        assert torch.equal(mine, ref), f"shift n={n} diverges from sglang"

    ple = pytest.importorskip("sglang.kernels.ops.qwen4_ple")
    ctx = torch.randint(0, VOCAB, (128, NGRAM_SIZE), device="cuda", dtype=torch.long, generator=g)
    if eos_frac:
        m = torch.rand(128, NGRAM_SIZE, device="cuda", generator=g) < eos_frac
        ctx = torch.where(m, torch.full_like(ctx, EOS), ctx)
    if not ple.can_fuse_qwen4_ngram_hash(ctx, mult, sizes, offs):
        pytest.skip("fused kernel declined this input")
    ref = ple.fused_qwen4_ngram_hash(ctx, mult, sizes, offs, EOS)
    mine = ngram_hash_ids(ctx, mult, sizes, offs, NGRAM_SIZE, HEADS_PER_NGRAM, EOS)
    assert torch.equal(mine, ref), "hash ids diverge from sglang fused kernel"


# ---- hyper-connection ----

HC_SHAPES = [(7, 64, 4, 16), (128, 2560, 4, 320)]
HC_DTYPES = [(torch.float32, 2e-5, 1e-5), (torch.bfloat16, 3e-2, 1e-2)]


def _hc_params(W, R, n, dtype, g):
    weight = (0.05 * torch.randn(W, device="cuda", dtype=dtype, generator=g)).requires_grad_()
    w_down = (torch.randn(R, W, device="cuda", dtype=dtype, generator=g) / W**0.5).requires_grad_()
    w_up = (torch.randn(W, R, device="cuda", dtype=dtype, generator=g) / R**0.5).requires_grad_()
    w_inj = (torch.randn(n, W, device="cuda", dtype=dtype, generator=g) / W**0.5).requires_grad_()
    return weight, w_down, w_up, w_inj


@pytest.mark.parametrize("T,C,n,R", HC_SHAPES)
@pytest.mark.parametrize("dtype,tol_mix,tol_comb", HC_DTYPES)
@pytest.mark.parametrize("with_inject", [True, False])
def test_hc_mix_inject(T, C, n, R, dtype, tol_mix, tol_comb, with_inject):
    g = torch.Generator(device="cuda").manual_seed(T)
    W, eps = n * C, 1e-6
    x = torch.randn(T, W, device="cuda", dtype=dtype, generator=g).requires_grad_()
    weight, w_down, w_up, w_inj = _hc_params(W, R, n, dtype, g)
    dmix = torch.randn(T, C, device="cuda", dtype=dtype, generator=g)
    dhp = torch.randn(T, n, device="cuda", dtype=torch.float32, generator=g)

    params = (x, weight, w_down, w_up) + ((w_inj,) if with_inject else ())
    normed = grouped_gemma_rmsnorm(x, weight, n, eps)
    mix_ref = hc_mix(normed, w_down, w_up, n, C, out_dtype=x.dtype)
    if with_inject:
        hp_ref = hc_inject_gate(normed, w_inj, n)
        torch.autograd.backward([mix_ref, hp_ref], [dmix, dhp])
    else:
        mix_ref.backward(dmix)
    ref_grads = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    mix_tri, hp_tri = hc_mix_inject_triton(x, weight, w_down, w_up, w_inj if with_inject else None, n, eps)
    if with_inject:
        torch.autograd.backward([mix_tri, hp_tri], [dmix, dhp])
        assert rel_err(hp_tri, hp_ref) < tol_mix
    else:
        mix_tri.backward(dmix)
    assert rel_err(mix_tri, mix_ref) < tol_mix
    for name, r, p in zip(["dx", "dw_norm", "dw_down", "dw_up", "dw_inj"], ref_grads, params, strict=False):
        err = rel_err(p.grad, r)
        assert err < tol_mix, f"{name}: {err:.2e} > {tol_mix}"


@pytest.mark.parametrize("T,C,n,R", HC_SHAPES)
@pytest.mark.parametrize("dtype,tol_mix,tol_comb", HC_DTYPES)
def test_hc_combine(T, C, n, R, dtype, tol_mix, tol_comb):
    g = torch.Generator(device="cuda").manual_seed(T + 1)
    W = n * C
    res = torch.randn(T, W, device="cuda", dtype=dtype, generator=g).requires_grad_()
    y = torch.randn(T, C, device="cuda", dtype=dtype, generator=g).requires_grad_()
    hp = torch.rand(T, n, device="cuda", dtype=torch.float32, generator=g).mul(2).requires_grad_()
    dout = torch.randn(T, W, device="cuda", dtype=dtype, generator=g)

    out_ref = hc_combine(res, y, hp, n, C)
    out_ref.backward(dout)
    ref_grads = [t.grad.clone() for t in (res, y, hp)]
    for t in (res, y, hp):
        t.grad = None

    out_tri = hc_combine_triton(res, y, hp, n)
    out_tri.backward(dout)
    assert rel_err(out_tri, out_ref) < tol_comb
    for name, r, t in zip(["dres", "dy", "dhpost"], ref_grads, (res, y, hp), strict=False):
        err = rel_err(t.grad, r)
        assert err < tol_comb, f"{name}: {err:.2e} > {tol_comb}"


def test_hc_mix_inject_3d_leading_shape():
    g = torch.Generator(device="cuda").manual_seed(99)
    x3 = torch.randn(17, 2, 4 * 64, device="cuda", dtype=torch.float32, generator=g)
    weight, w_down, w_up, w_inj = _hc_params(4 * 64, 16, 4, torch.float32, g)
    m3, hp3 = hc_mix_inject_triton(x3, weight, w_down, w_up, w_inj, 4, 1e-6)
    assert m3.shape == (17, 2, 64) and hp3.shape == (17, 2, 4)


# ---- PLE gate + conv ----

# fp32 tol is looser than HC/QSA: d(gate)/d(score) ~ 1/(2*sqrt(|s|)) blows up
# toward the 1e-6 clamp knee and amplifies summation-order differences.
PLE_CASES = [
    (64, 64, 4, [0, 5, 6, 30, 64]),
    (128, 2560, 4, [0, 1, 3, 70, 128]),
]


@pytest.mark.parametrize("T,C,n,segs", PLE_CASES)
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 5e-4), (torch.bfloat16, 4e-2)])
def test_ple_gate_conv(T, C, n, segs, dtype, tol):
    g = torch.Generator(device="cuda").manual_seed(T)
    W, eps, K, dil = n * C, 1e-6, 4, 3

    def mk(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)

    hc = mk(T, W).requires_grad_()
    key = mk(T, W).requires_grad_()
    value = mk(T, C).requires_grad_()
    wk = (0.05 * mk(W)).requires_grad_()
    wq = (0.05 * mk(W)).requires_grad_()
    wc = (0.05 * mk(W)).requires_grad_()
    convw = (mk(W, 1, K) / K).requires_grad_()
    cu = torch.tensor(segs, dtype=torch.int32, device="cuda")
    dout = mk(T, W)

    params = (hc, key, value, wk, wq, wc, convw)
    ref = ple_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu)
    ref.backward(dout)
    ref_grads = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    tri = ple_gate_conv_triton(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu)
    tri.backward(dout)
    assert rel_err(tri, ref) < tol
    names = ["dhc", "dkey", "dvalue", "dwk", "dwq", "dwc", "dconvw"]
    for name, r, p in zip(names, ref_grads, params, strict=False):
        err = rel_err(p.grad, r)
        assert err < tol, f"{name}: {err:.2e} > {tol}"


# ---- QSA sparse attention ----

QSA_CASES = [
    (128, 128, 4, 2, 64, 32, torch.float32),
    (257, 257, 6, 2, 128, 64, torch.float32),
    (515, 515, 24, 2, 128, 96, torch.bfloat16),
    (700, 700, 8, 8, 64, 50, torch.float32),
]


@pytest.mark.parametrize("T,S,Hq,Hkv,D,K,dtype", QSA_CASES)
def test_qsa_sparse_attention(T, S, Hq, Hkv, D, K, dtype):
    g = torch.Generator(device="cuda").manual_seed(T)
    q = torch.randn(T, Hq, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    k = torch.randn(S, Hkv, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    v = torch.randn(S, Hkv, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    # Unique indices per row: production selections are unique by construction,
    # and the kernel is list-semantics (a duplicate would be counted twice,
    # which the mask-based reference cannot represent).
    idx = torch.rand(T, S, device="cuda", generator=g).topk(K, dim=-1).indices.to(torch.int32)
    keep = torch.rand(T, K, device="cuda", generator=g) > 0.3
    keep[:, 0] = True
    idx = torch.where(keep, idx, torch.full_like(idx, -1))

    scale = D**-0.5
    out_t = qsa_sparse_attention_triton(q, k, v, idx, scale)
    gout = torch.randn_like(out_t)
    out_t.backward(gout)
    grads_t = [q.grad.clone(), k.grad.clone(), v.grad.clone()]

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_r = qsa_reference(q2, k2, v2, idx, scale).to(dtype)
    out_r.backward(gout)

    tol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    assert rel_err(out_t, out_r) < tol
    for name, t, r in zip(["dq", "dk", "dv"], grads_t, (q2.grad, k2.grad, v2.grad), strict=False):
        err = rel_err(t, r)
        assert err < tol, f"{name}: {err:.2e} > {tol}"
