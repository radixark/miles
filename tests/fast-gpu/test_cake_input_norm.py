"""--linear-attention-input-norm cake: the zero-centred input RMSNorm of the GDN layers on the Cake kernels of the
miles-ext-kernels submodule (miles_ext_kernels.rmsnorm).

At the Qwen3.5 widths the kernels serve, both variants match a float64 reference within the bounds the TE norm meets
(checked here too), and det repeats bit for bit. The cake norm holds the TE norm's single ``weight`` (w of the
``(1 + w)`` scale, zeros at init) and loads its state dict; a hidden size the kernels do not serve (Qwen3.5-4B's
2560) is refused when the norm is built.

    python tests/fast-gpu/test_cake_input_norm.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the Cake RMSNorm kernels run on SM100a / SM103a",
)

from miles_plugins.models.qwen3_5 import gdn_input_layernorm  # noqa: E402

EPS = 1e-6
TOKENS = 8192
HIDDEN_SIZES = [2048, 4096]
# relative L2 error against float64, the bounds test_linear_attn_input_norm.py sets for the te and hf norms
LIMITS = {"out": 4e-3, "dx": 5e-3, "dweight": 5e-3}


def _args(variant: str) -> SimpleNamespace:
    return SimpleNamespace(miles_ext_kernels_args={"cake_rmsnorm_variant": variant})


def _inputs(hidden, tokens=TOKENS):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", generator=g)
    x = x * torch.exp(torch.randn(hidden, device="cuda", generator=g))
    x = x * torch.exp(1.5 * torch.randn(tokens, 1, device="cuda", generator=g))
    weight = (0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    grad_out = (0.1 * torch.randn(tokens, hidden, device="cuda", generator=g)).bfloat16()
    return x.bfloat16().unsqueeze(1), weight, grad_out.unsqueeze(1)


def _norm(kind, hidden, weight, variant="det"):
    norm = gdn_input_layernorm(kind, hidden, EPS, torch.bfloat16, args=_args(variant)).to("cuda")
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _run(norm, x, grad_out):
    norm.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out = norm(x)
    out.backward(grad_out)
    torch.cuda.synchronize()
    return out.detach(), x.grad, norm.weight.grad


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_cake_input_norm_is_as_accurate_as_te(variant, hidden):
    x, weight, grad_out = _inputs(hidden)
    x64, w64 = x.double().requires_grad_(), weight.double().requires_grad_()
    ref = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w64)
    ref.backward(grad_out.double())
    expected = (ref.detach(), x64.grad, w64.grad)

    for kind in ("te", "cake"):
        out, dx, dweight = _run(_norm(kind, hidden, weight, variant), x, grad_out)

        assert out.dtype == torch.bfloat16 and dweight.dtype == torch.bfloat16
        values = zip(LIMITS, (out, dx, dweight), expected, strict=True)
        errors = {name: _rel(value, reference) for name, value, reference in values}
        assert all(errors[name] < limit for name, limit in LIMITS.items()), (kind, errors)


def test_det_input_norm_repeats_bit_for_bit():
    x, weight, grad_out = _inputs(HIDDEN_SIZES[0])
    norm = _norm("cake", HIDDEN_SIZES[0], weight)

    first, second = _run(norm, x, grad_out), _run(norm, x, grad_out)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_the_cake_norm_has_the_te_parameter_surface():
    hidden = HIDDEN_SIZES[0]
    te_norm = gdn_input_layernorm("te", hidden, EPS, torch.bfloat16)
    cake_norm = gdn_input_layernorm("cake", hidden, EPS, torch.bfloat16, args=_args("det"))
    assert list(cake_norm.state_dict()) == list(te_norm.state_dict()) == ["weight"]
    assert not cake_norm.weight.any()

    x, weight, _ = _inputs(hidden, tokens=64)
    te_norm = _norm("te", hidden, weight)
    cake_norm.load_state_dict(te_norm.state_dict())
    with torch.no_grad():
        te_out, cake_out = te_norm(x).float(), cake_norm(x).float()
    assert ((te_out - cake_out).abs() <= te_out.abs() * 2**-7).all()


def test_a_hidden_size_the_kernels_do_not_serve_is_refused_when_built():
    with pytest.raises(ValueError, match="not 2560"):
        gdn_input_layernorm("cake", 2560, EPS, torch.bfloat16, args=_args("det"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
