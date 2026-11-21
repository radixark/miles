"""Linear-attention kernel selection, and fla vs flashqla numerical equivalence for GDN.

The equivalence tests need a Hopper (SM90+) GPU with both `fla` and `flash_qla`; they skip otherwise.
"""

import os
import sys
import types

import pytest

from miles_plugins.models import linear_attn


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        linear_attn.gdn_kernel("nope")


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


@pytest.mark.parametrize("kernel_built", [True, False])
def test_short_conv_backend_prefers_the_miles_backward_when_built(monkeypatch, kernel_built):
    monkeypatch.delenv("FLA_CONV_BACKEND", raising=False)
    monkeypatch.setattr(linear_attn, "conv_bwd_kernel_available", lambda: kernel_built)
    expected = "miles" if kernel_built else linear_attn._fla_short_conv_backend()
    assert linear_attn.short_conv_backend.__wrapped__() == expected


def _seqtile_args(**changes):
    """A shape the "miles" backward takes (8192 x 1024, above its 1M-element size floor), with ``changes``.
    Meta tensors: the gate reads shapes and strides only."""
    torch = pytest.importorskip("torch")
    args = {
        "x": torch.empty(1, 8192, 1024, dtype=torch.bfloat16, device="meta"),
        "weight": torch.empty(1024, 4, device="meta"),
        "activation": "silu",
        "cu_seqlens": torch.empty(5, dtype=torch.int32, device="meta"),
        "cp_context": None,
    }
    return {**args, **changes}


_SEQTILE_FALLBACKS = {
    "below_floor": lambda torch: {"x": torch.empty(1, 512, 1024, dtype=torch.bfloat16, device="meta")},
    # 16-token sequences, under half a 64-row tile on average
    "short_sequences": lambda torch: {"cu_seqlens": torch.empty(8192 // 16 + 1, dtype=torch.int32, device="meta")},
    "context_parallel": lambda torch: {"cp_context": object()},
    "no_activation": lambda torch: {"activation": None},
    "no_boundaries": lambda torch: {"cu_seqlens": None},
    "int64_boundaries": lambda torch: {"cu_seqlens": torch.empty(5, dtype=torch.int64, device="meta")},
    "fp16": lambda torch: {"x": torch.empty(1, 8192, 1024, dtype=torch.float16, device="meta")},
    "width_3": lambda torch: {"weight": torch.empty(1024, 3, device="meta")},
    # rows 1028 channels apart, starting 4 channels in: not 16-byte aligned
    "unaligned_rows": lambda torch: {"x": torch.empty(1, 8192, 1028, dtype=torch.bfloat16, device="meta")[..., 4:]},
}


def test_seqtile_applies_on_the_supported_shape():
    assert linear_attn._seqtile_applies(**_seqtile_args())


@pytest.mark.parametrize("case", list(_SEQTILE_FALLBACKS))
def test_seqtile_falls_back(case):
    torch = pytest.importorskip("torch")
    assert not linear_attn._seqtile_applies(**_seqtile_args(**_SEQTILE_FALLBACKS[case](torch)))


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
