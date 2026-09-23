"""Numerical equivalence of the Qwen GDN backends.

* fla vs flashqla: needs a Hopper (SM90+) GPU with both `fla` and `flash_qla`; skips otherwise.
* fla vs loom (deterministic generated kernels): needs a Blackwell SM100a/SM103a GPU and `fla`;
  also checks that the loom forward and backward are bit-identical across calls.
"""

import importlib.util
from pathlib import Path

import pytest


def load_backend_module():
    module_path = Path(__file__).resolve().parents[2] / "miles_plugins" / "models" / "qwen_gdn_backend.py"
    spec = importlib.util.spec_from_file_location("test_qwen_gdn_backend_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unknown_backend_raises_value_error():
    module = load_backend_module()
    with pytest.raises(ValueError, match="Unsupported Qwen GDN backend"):
        module.get_chunk_gated_delta_rule("nope")


def test_loom_backend_routes_to_the_generated_kernels():
    pytest.importorskip("torch")
    module = load_backend_module()
    fn = module.get_chunk_gated_delta_rule("loom")
    assert fn.__module__ == "miles_plugins.models.gdn_chunk_train.ops"
    assert fn.__name__ == "chunk_gated_delta_rule"


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
    module = load_backend_module()

    fla_kernel = module.get_chunk_gated_delta_rule("fla")
    flashqla_kernel = module.get_chunk_gated_delta_rule("flashqla")

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
        pytest.skip("the deterministic GDN kernels need an SM100a/SM103a GPU")
    pytest.importorskip("fla.ops.gated_delta_rule")
    return torch


def _autograd_step(torch, kernel, inputs, cu, use_cu: bool):
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
        cu_seqlens=cu if use_cu else None,
    )
    torch.manual_seed(1)
    dout = torch.randn_like(out)
    out.backward(dout)
    return {"out": out.detach(), "dq": query.grad, "dk": key.grad, "dv": value.grad, "dg": g.grad, "dbeta": beta.grad}


@pytest.mark.parametrize("use_cu", [True, False])
def test_fla_loom_equivalence_and_determinism(use_cu):
    torch = _require_loom()
    module = load_backend_module()
    fla_kernel = module.get_chunk_gated_delta_rule("fla")
    loom_kernel = module.get_chunk_gated_delta_rule("loom")

    query, key, value, g, beta, cu = _make_inputs(torch, torch.bfloat16, device="cuda")
    if not use_cu:
        # equal-length batch of 2 sequences instead of the packed varlen layout
        total = query.shape[1] // 2 * 2
        query, key, value, g, beta = (
            t[:, :total].reshape(2, total // 2, *t.shape[2:]) for t in (query, key, value, g, beta)
        )
    beta = beta.to(torch.bfloat16)  # as the model feeds it: sigmoid(b) in the activation dtype
    inputs = (query, key, value, g, beta)

    ref = _autograd_step(torch, fla_kernel, inputs, cu, use_cu)
    first = _autograd_step(torch, loom_kernel, inputs, cu, use_cu)
    second = _autograd_step(torch, loom_kernel, inputs, cu, use_cu)
    for name in ref:
        assert torch.equal(first[name], second[name]), f"loom {name} is not bit-deterministic"
        a, b = first[name].float(), ref[name].float()
        scale = max(1.0, b.abs().max().item()) if name in ("dg",) else 1.0
        torch.testing.assert_close(a, b, atol=1e-2 * scale, rtol=1e-2, msg=lambda m, name=name: f"{name}: {m}")
