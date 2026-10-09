"""Linear-attention kernel selection, and fla vs flashqla / fla vs loom numerical equivalence for GDN.

The flashqla equivalence tests need a Hopper (SM90+) GPU with both `fla` and `flash_qla`; the loom ones a
Blackwell (SM100a / SM103a) GPU with `fla`; they skip otherwise.
"""

import os
import sys
import types

import pytest

from miles_plugins.models import linear_attn


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        linear_attn.gdn_kernel("nope")


def test_loom_backend_routes_to_the_generated_kernels():
    pytest.importorskip("torch")
    fn = linear_attn.gdn_kernel("loom")
    assert fn.__module__ == "miles_plugins.models.gdn_chunk_train.ops"
    assert fn.__name__ == "chunk_gated_delta_rule"


@pytest.mark.parametrize(
    "capability,user_value,expected", [((10, 0), None, "0"), ((9, 0), None, None), ((10, 0), "1", "1")]
)
def test_kda_uses_the_triton_backward_on_blackwell_unless_set(monkeypatch, capability, user_value, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "fla.ops.kda", types.SimpleNamespace(chunk_kda="chunk_kda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    if user_value is not None:
        monkeypatch.setenv("FLA_TILELANG", user_value)
    assert linear_attn.kda_kernel.__wrapped__() == "chunk_kda"
    assert os.environ.get("FLA_TILELANG") == expected


def test_short_conv_backend_respects_fla_conv_backend(monkeypatch):
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    assert linear_attn.short_conv_backend.__wrapped__() == "cuda"


NUM_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
SEQLENS = [128, 256, 128]


def _require_backends():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required to compare GDN kernels")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FlashQLA requires NVIDIA SM90 (Hopper) or newer")
    pytest.importorskip("fla.ops.gated_delta_rule")
    pytest.importorskip("flash_qla")
    return torch


def _make_inputs(torch, dtype, device):
    torch.manual_seed(0)
    total = sum(SEQLENS)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    query = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    key = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    value = randn(1, total, NUM_HEADS, HEAD_V_DIM)
    # g: per-head log-decay (<= 0); beta: gate in (0, 1) -- as the model feeds them.
    g = -torch.nn.functional.softplus(randn(1, total, NUM_HEADS).float())
    beta = randn(1, total, NUM_HEADS).float().sigmoid()

    cu = torch.tensor([0, *SEQLENS], device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
    return query, key, value, g, beta, cu


def _run(kernel, query, key, value, g, beta, cu_seqlens):
    out, _ = kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    return out


# FlashQLA only supports half precision (asserts on float32).
@pytest.mark.parametrize(
    "dtype_name, atol, rtol",
    [
        ("bfloat16", 4e-2, 4e-2),
        ("float16", 1e-2, 1e-2),
    ],
)
def test_fla_flashqla_equivalence(dtype_name, atol, rtol):
    torch = _require_backends()
    dtype = getattr(torch, dtype_name)

    fla_kernel = linear_attn.gdn_kernel("fla")
    flashqla_kernel = linear_attn.gdn_kernel("flashqla")

    query, key, value, g, beta, cu = _make_inputs(torch, dtype, device="cuda")

    fla_out = _run(fla_kernel, query, key, value, g, beta, cu).float()
    flashqla_out = _run(flashqla_kernel, query, key, value, g, beta, cu).float()

    assert fla_out.shape == flashqla_out.shape
    diff = (fla_out - flashqla_out).abs()
    denom = fla_out.abs().max().item() + 1e-6
    print(
        f"\n[GDN fla vs flashqla] dtype={dtype_name} "
        f"max_abs_diff={diff.max().item():.3e} mean_abs_diff={diff.mean().item():.3e} "
        f"max_rel_diff={diff.max().item() / denom:.3e}"
    )

    torch.testing.assert_close(flashqla_out, fla_out, atol=atol, rtol=rtol)


def _require_loom():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required to compare GDN kernels")
    from miles_plugins.models.gdn_chunk_train import SUPPORTED_CAPABILITIES

    if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
        pytest.skip("the deterministic GDN kernels need an SM100a / SM103a GPU")
    pytest.importorskip("fla.ops.gated_delta_rule")
    return torch


def _grouped_inputs(torch, v_per_k: int, use_cu: bool):
    """q/k with NUM_HEADS key heads, v / g / beta with ``v_per_k`` value heads per key head, as the head-sharded
    layer feeds the kernel; packed varlen (``cu_seqlens``) or an equal-length batch of two sequences."""
    torch.manual_seed(0)
    total = sum(SEQLENS)
    num_v_heads = NUM_HEADS * v_per_k

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    query = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    key = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    value = randn(1, total, num_v_heads, HEAD_V_DIM)
    g = -torch.nn.functional.softplus(randn(1, total, num_v_heads, dtype=torch.float32))  # fp32 log-decay <= 0
    beta = randn(1, total, num_v_heads).sigmoid()  # sigmoid(b) in the activation dtype
    cu_seqlens = torch.tensor([0, *SEQLENS], device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    if not use_cu:
        query, key, value, g, beta = (t.reshape(2, total // 2, *t.shape[2:]) for t in (query, key, value, g, beta))
        cu_seqlens = None
    return (query, key, value, g, beta), cu_seqlens


def _autograd_step(torch, kernel, inputs, cu_seqlens):
    query, key, value, g, beta = (t.detach().clone().requires_grad_(True) for t in inputs)
    out, _ = kernel(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    torch.manual_seed(1)
    out.backward(torch.randn_like(out))
    return {"out": out.detach(), "dq": query.grad, "dk": key.grad, "dv": value.grad, "dg": g.grad, "dbeta": beta.grad}


@pytest.mark.parametrize("v_per_k", [1, 2])
@pytest.mark.parametrize("use_cu", [True, False])
def test_fla_loom_equivalence_and_determinism(use_cu, v_per_k):
    torch = _require_loom()
    fla_kernel = linear_attn.gdn_kernel("fla")
    loom_kernel = linear_attn.gdn_kernel("loom")
    inputs, cu_seqlens = _grouped_inputs(torch, v_per_k, use_cu)

    ref = _autograd_step(torch, fla_kernel, inputs, cu_seqlens)
    first = _autograd_step(torch, loom_kernel, inputs, cu_seqlens)
    second = _autograd_step(torch, loom_kernel, inputs, cu_seqlens)
    for name in ref:
        assert torch.equal(first[name], second[name]), f"loom {name} is not bit-deterministic"
        actual, expected = first[name].float(), ref[name].float()
        scale = max(1.0, expected.abs().max().item()) if name == "dg" else 1.0
        torch.testing.assert_close(
            actual, expected, atol=1e-2 * scale, rtol=1e-2, msg=lambda m, name=name: f"{name}: {m}"
        )
