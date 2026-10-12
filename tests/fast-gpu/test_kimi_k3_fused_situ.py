"""The fused Kimi-K3 SiTU-and-mul (`--kimi-k3-situ-backend fused`) against the eager op: the forward is
bit-identical on almost every element and within one bf16 ulp everywhere; the closed-form backward differs from
autograd's fp32 rounding on a small fraction of elements (within 2 ulp on > 99.9 %, observed 99.987 %). Both are
run-to-run identical."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models.kimi_k3.fused_situ import fused_situ_and_mul  # noqa: E402
from miles_plugins.models.kimi_k3.ops import situ_and_mul  # noqa: E402


def _ulps(ref: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    ref, got = ref.float(), got.float()
    one_ulp = torch.where(ref == 0, torch.full_like(ref, 2.0**-133), ref.abs() * 2.0**-7)
    return (ref - got).abs() / one_ulp


def _run(fn, x, grad_out):
    x = x.clone().requires_grad_()
    y = fn(x)
    y.backward(grad_out)
    return y.detach(), x.grad


# expert intermediates of the K3 trainer: tokens x top-k rows of [gate | linear], 3072 each
@pytest.mark.parametrize("rows,inter", [(1000, 3072), (16384, 3072), (4096, 6144)])
def test_fused_matches_eager(rows, inter):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 2 * inter, device="cuda", generator=g) * 3).to(torch.bfloat16)
    grad_out = torch.randn(rows, inter, device="cuda", generator=g).to(torch.bfloat16)

    y_ref, gx_ref = _run(situ_and_mul, x, grad_out)
    y, gx = _run(fused_situ_and_mul, x, grad_out)

    assert y.dtype == y_ref.dtype and gx.dtype == gx_ref.dtype
    assert (y == y_ref).float().mean().item() > 0.9999
    assert _ulps(y_ref, y).max().item() <= 1.0
    assert (gx == gx_ref).float().mean().item() > 0.999
    assert (_ulps(gx_ref, gx) <= 2.0).float().mean().item() > 0.999
    rel_l2 = ((gx.float() - gx_ref.float()).norm() / gx_ref.float().norm()).item()
    assert rel_l2 < 1e-4


def test_fused_is_deterministic_and_handles_strided_and_empty_inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    base = (torch.randn(512, 2 * 3072 + 64, device="cuda", generator=g) * 3).to(torch.bfloat16)
    x = base[:, : 2 * 3072]  # row-strided view
    grad_out = torch.randn(512, 3072, device="cuda", generator=g).to(torch.bfloat16)

    y0, gx0 = _run(fused_situ_and_mul, x, grad_out)
    y1, gx1 = _run(fused_situ_and_mul, x.contiguous(), grad_out)
    assert torch.equal(y0, y1) and torch.equal(gx0, gx1)

    empty = torch.empty(0, 2 * 3072, device="cuda", dtype=torch.bfloat16)
    assert fused_situ_and_mul(empty).shape == (0, 3072)


def test_backend_selection_configures_the_activation():
    from types import SimpleNamespace

    from miles_plugins.models.kimi_k3.model import configure_kimi_k3

    for backend, fn in (("eager", situ_and_mul), ("fused", fused_situ_and_mul)):
        config = SimpleNamespace(num_layers=4, num_attention_heads=96, moe_latent_size=3584)
        configure_kimi_k3(config, situ_backend=backend)
        assert config.gated_activation_func is fn
