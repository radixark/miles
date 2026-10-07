"""Linear-attention kernel selection, and fla vs flashqla numerical equivalence for GDN.

The equivalence tests need a Hopper (SM90+) GPU with both `fla` and `flash_qla`; they skip otherwise.
"""

import pytest

from miles_plugins.models.linear_attn import gdn_kernel


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        gdn_kernel("nope")


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

    fla_kernel = gdn_kernel("fla")
    flashqla_kernel = gdn_kernel("flashqla")

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
