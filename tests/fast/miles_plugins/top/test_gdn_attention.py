"""The Qwen3.5 / Qwen3.8 full attention's Q/K norm + partial RoPE + gate split: serving's fused kernel gives the value,
the trainer's unfused chain (serving's norm with its backward, then the FP32 rotation) the gradient."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

gdn_attention = pytest.importorskip("miles_plugins.top.gdn_attention")
from miles_plugins.top import gdn_norm  # noqa: E402
from miles_plugins.top.gdn_norm import TopGemmaRMSNorm  # noqa: E402
from miles_plugins.top.kernels import surrogate  # noqa: E402

T, HEADS, KV_HEADS, D, ROTARY = 5, 4, 2, 16, 8


class _GemmaNormMath:
    """Serving's norm kernel and its TE backward, as the PyTorch math they compute (CPU)."""

    @staticmethod
    def apply(x, weight, norm):
        xf = x.float()
        inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.variance_epsilon)
        return (xf * inv * (1.0 + weight.float())).to(x.dtype)


def _rope(dtype=torch.float32):
    g = torch.Generator().manual_seed(1)
    return SimpleNamespace(
        cos_sin_cache=torch.randn(64, ROTARY, generator=g).to(dtype),
        head_size=D,
        rotary_dim=ROTARY,
        mrope_section=None,
        axis_map=None,
    )


def _norm(seed):
    norm = TopGemmaRMSNorm(SimpleNamespace(), D, 1e-6, role="q_layernorm")
    with torch.no_grad():
        norm.weight.copy_(torch.randn(D, generator=torch.Generator().manual_seed(seed)))
    return norm


def _unfused(qg, k, wq, wk, *, q_norm, k_norm, positions, rope):
    q, gate = qg.reshape(T, HEADS, 2 * D).chunk(2, -1)
    q = _GemmaNormMath.apply(q.reshape(-1, D), wq, q_norm._sgl).reshape(T, -1)
    k = _GemmaNormMath.apply(k.reshape(-1, D), wk, k_norm._sgl).reshape(T, -1)
    rotate = gdn_attention.rotary_reference
    return rotate(q, positions, rope), rotate(k, positions, rope), gate.reshape(T, -1)


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    monkeypatch.setattr(surrogate, "reference_mode", contextlib.nullcontext)
    monkeypatch.setattr(gdn_norm, "_GemmaNorm", _GemmaNormMath)


def test_the_gradient_is_the_unfused_chains_and_the_value_the_kernels(monkeypatch):
    """Every input, the gate half of qg and both norm weights included, takes the unfused chain's gradient exactly;
    the forward returns the fused kernel's outputs with the gate flattened to [T, heads * D]."""
    q_norm, k_norm, rope = _norm(2), _norm(3), _rope()
    table = gdn_attention.keep_fp32_cos_sin(rope)
    rope.cos_sin_cache = rope.cos_sin_cache.bfloat16()  # what the trainer's BF16 model cast does to the buffer
    positions = torch.tensor([0, 1, 2, 7, 9])
    kernel_out = (torch.randn(T, HEADS * D), torch.randn(T, KV_HEADS * D), torch.randn(T, HEADS, D))

    def fused_kernel(qg, k, wq, wk, cos_sin_cache, pos, eps, heads, kv_heads, head_dim, rotary_dim, **kwargs):
        assert (heads, kv_heads, head_dim, rotary_dim, eps) == (HEADS, KV_HEADS, D, ROTARY, 1e-6)
        assert cos_sin_cache is table and kwargs == dict(has_gate=True, mrope_axis_map=None)
        return kernel_out

    monkeypatch.setattr(gdn_attention, "fused_qk_gemma_rmsnorm_rope_gate", fused_kernel)
    g = torch.Generator().manual_seed(0)
    qg = torch.randn(T, HEADS * 2 * D, generator=g, requires_grad=True)
    k = torch.randn(T, KV_HEADS * D, generator=g, requires_grad=True)
    out = gdn_attention.fused_qk_norm_rope_gate(
        qg, k, q_norm=q_norm, k_norm=k_norm, positions=positions, rope=rope, heads=HEADS, kv_heads=KV_HEADS
    )
    assert all(torch.equal(o, v.reshape(T, -1)) for o, v in zip(out, kernel_out, strict=True))

    grads = [torch.randn(o.shape, generator=g) for o in out]
    got = torch.autograd.grad(out, (qg, k, q_norm.weight, k_norm.weight), grads)
    leaves = [t.detach().requires_grad_() for t in (qg, k, q_norm.weight, k_norm.weight)]
    ref = _unfused(*leaves, q_norm=q_norm, k_norm=k_norm, positions=positions, rope=rope)
    want = torch.autograd.grad(ref, leaves, grads)
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


def test_a_rounded_cos_sin_table_is_refused():
    """The fused kernel would read a BF16 table without complaint; serving's reads the FP32 CPU-derived one. A table
    built BF16 cannot be kept, and the fused op refuses a rope whose FP32 table was never kept."""
    with pytest.raises(ValueError, match=r"\[top\].*built FP32"):
        gdn_attention.keep_fp32_cos_sin(_rope(torch.bfloat16))
    qg, k = torch.zeros(T, HEADS * 2 * D), torch.zeros(T, KV_HEADS * D)
    with pytest.raises(ValueError, match=r"\[top\].*FP32 table"):
        gdn_attention.fused_qk_norm_rope_gate(
            qg,
            k,
            q_norm=_norm(2),
            k_norm=_norm(3),
            positions=torch.arange(T),
            rope=_rope(),
            heads=HEADS,
            kv_heads=KV_HEADS,
        )
