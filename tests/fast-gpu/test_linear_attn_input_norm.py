"""The TE input RMSNorm of the GDN layer matches the eager HF one and keeps its single `weight` parameter."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models.qwen3_5 import gdn_input_layernorm  # noqa: E402

HIDDEN, EPS = 2048, 1e-6


def _inputs(tokens):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, HIDDEN, device="cuda", generator=g)
    x = x * torch.exp(torch.randn(HIDDEN, device="cuda", generator=g))
    x = x * torch.exp(1.5 * torch.randn(tokens, 1, device="cuda", generator=g))
    weight = (0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).bfloat16()
    grad_out = (0.1 * torch.randn(tokens, HIDDEN, device="cuda", generator=g)).bfloat16()
    return x.bfloat16(), weight, grad_out


def _norm(kind, weight):
    norm = gdn_input_layernorm(kind, HIDDEN, EPS, torch.bfloat16).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _run(kind, x, weight, grad_out):
    norm = _norm(kind, weight)
    x = x.clone().requires_grad_()
    out = norm(x)
    out.backward(grad_out)
    return out, x.grad, norm.weight.grad


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def test_matches_hf_and_float64():
    x, weight, grad_out = _inputs(8192)
    x64, w64 = x.double().requires_grad_(), weight.double().requires_grad_()
    ref = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w64)
    ref.backward(grad_out.double())
    results = {kind: _run(kind, x, weight, grad_out) for kind in ("te", "hf")}
    for kind, (out, dx, dw) in results.items():
        assert out.dtype == torch.bfloat16 and dw.dtype == torch.bfloat16
        assert _rel(out, ref.detach()) < 4e-3, kind
        assert _rel(dx, x64.grad) < 5e-3, kind
        assert _rel(dw, w64.grad) < 5e-3, kind
    (te_out, te_dx, te_dw), (hf_out, hf_dx, hf_dw) = results["te"], results["hf"]
    differ = te_out.view(torch.int16) != hf_out.view(torch.int16)
    assert differ.float().mean().item() < 1e-4
    ulps = (te_out.view(torch.int16).int() - hf_out.view(torch.int16).int()).abs()
    assert ulps.max().item() <= 1
    assert abs(_rel(te_dx, x64.grad) - _rel(hf_dx, x64.grad)) < 1e-6
    assert abs(_rel(te_dw, w64.grad) - _rel(hf_dw, w64.grad)) < 1e-6


def test_same_parameter_surface():
    te_norm, hf_norm = (gdn_input_layernorm(kind, HIDDEN, EPS, torch.bfloat16) for kind in ("te", "hf"))
    assert list(te_norm.state_dict()) == list(hf_norm.state_dict()) == ["weight"]
    assert not te_norm.weight.any() and not hf_norm.weight.any()
    x, weight, _ = _inputs(64)
    hf_norm = _norm("hf", weight)
    te_norm = gdn_input_layernorm("te", HIDDEN, EPS, torch.bfloat16).to("cuda")
    te_norm.load_state_dict(hf_norm.state_dict())
    with torch.no_grad():
        te_out, hf_out = te_norm(x).float(), hf_norm(x).float()
    assert ((te_out - hf_out).abs() <= hf_out.abs() * 2**-7).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
