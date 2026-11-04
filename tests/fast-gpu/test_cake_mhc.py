"""CakeHyperConnection (``--mhc-backend cake``) against Megatron's HyperConnectionModule and the FP64 function.

One GLM-5.3 connection (hidden 4096, 4 streams; Megatron's module with GLM-5.3's patched RMS normalisation) of
``s = 64, b = 2`` tokens around a stand-in branch ``x = U * w``:

* both modules' ``U``, ``Y``, ``dX`` and parameter gradients within BF16-coefficient tolerances (relative L2) of the
  FP64 function: Megatron applies BF16-cast coefficients, the Cake kernels FP32 ones;
* ``det`` bitwise run to run, and under Megatron's ``tensor_parallel.checkpoint``;
* ``fast``: ``U``, ``Y``, ``dX`` and the ``alpha_*`` / ``bias`` gradients bitwise the ``det`` ones, the mapping
  gradient within FP32 summation-order noise;
* the GLM-5.3 head bit for bit ``HyperConnectionModule.output_contract`` and its gradient;
* ``apply_mhc_backend`` with the real kernel package.

The kernels' own suites (both variants against FP64 at the miles hidden sizes, the head, the variant rules) run in
``3rdparty/miles-ext-kernels/tests/mhc`` through ``tests/fast-gpu/kernels/test_miles_ext_kernels.py``.
"""

import argparse
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the Cake mHC kernels run on NVIDIA SM100a / SM103a",
)

# a missing submodule fails the collection: on the Blackwell runners a skip would hide the coverage
import miles_ext_kernels.mhc as mhc  # noqa: E402
from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.transformer.hyper_connection import HyperConnectionModule  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from miles_ext_kernels.mhc import ops  # noqa: E402

from miles_plugins.models.cake_hyper_connection import (  # noqa: E402
    CakeHyperConnection,
    CakeMeanOutputContraction,
    apply_mhc_backend,
)

S, B, N, H = 64, 2, 4, 4096
RMS_EPS = 1e-5  # GLM-5.3's layernorm_epsilon, the epsilon its patched normalisation uses
F64 = torch.float64
# relative L2 against the FP64 function, for both modules
TOLERANCES = {
    "U": 1e-2,
    "Y": 1e-2,
    "dX": 2e-2,
    "mapping_proj.weight": 3e-2,
    "bias": 3e-2,
    "alpha_pre": 5e-2,
    "alpha_post": 5e-2,
    "alpha_res": 5e-2,
}


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=32,
        params_dtype=torch.bfloat16,
        bf16=True,
        enable_hyper_connections=True,
        num_residual_streams=N,
        layernorm_epsilon=RMS_EPS,
        hidden_dropout=0.0,
    )


def _glm_proj_rms(x, weight, eps):
    """``glm5_next._reference_proj_rms``: GLM-5.3's projection and RMS normalisation of the mapping input."""
    return torch.matmul(x, weight.t()), torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _built(cls, params: dict) -> torch.nn.Module:
    """A connection with the parameters ``params`` and what ``glm5_next._patch_reference_proj_rms`` sets on it."""
    module = cls(_config(), 1).cuda()
    module.norm_eps = RMS_EPS
    module._proj_rms_op = _glm_proj_rms
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(params[name])
    return module


@pytest.fixture(autouse=True)
def variant(monkeypatch):
    """Set the Cake variant of the next calls through the package's configure; restored after the test."""
    monkeypatch.setattr(ops, "VARIANT", ops.VARIANT)

    def select(name: str) -> None:
        mhc.configure(argparse.Namespace(cake_mhc_variant=name))

    select("det")
    return select


@pytest.fixture(scope="module")
def params() -> dict:
    """Megatron's initialisation of the mapping, with non-trivial scales and offsets."""
    torch.manual_seed(0)
    module = HyperConnectionModule(_config(), 1)
    with torch.no_grad():
        for alpha in (module.alpha_pre, module.alpha_post, module.alpha_res):
            alpha.copy_(1.0 + 0.1 * torch.randn_like(alpha))
        module.bias.copy_(0.1 * torch.randn_like(module.bias))
    return {name: parameter.detach().cuda() for name, parameter in module.named_parameters()}


@pytest.fixture(scope="module")
def inputs() -> dict:
    g = torch.Generator(device="cuda").manual_seed(2032)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    return {"hidden_states": randn(S, B, N * H), "w": randn(S, B, H), "dY": randn(S, B, N * H)}


def _layer(module, hidden_states, w):
    U, h_res, h_post, residual = module(hidden_states)
    return U, module.fused_h_res_h_post_bda(h_res, residual, h_post, (U * w, None), 0.0, True, False)


def _run(module, inputs: dict, *, checkpointed: bool = False) -> dict:
    module.zero_grad(set_to_none=True)
    hidden_states = inputs["hidden_states"].clone().requires_grad_(True)
    if checkpointed:
        U = None
        Y = tensor_parallel.checkpoint(lambda x: _layer(module, x, inputs["w"])[1], False, hidden_states)
    else:
        U, Y = _layer(module, hidden_states, inputs["w"])
    torch.autograd.backward([Y], [inputs["dY"]])
    out = {"U": None if U is None else U.detach(), "Y": Y.detach(), "dX": hidden_states.grad}
    out.update({name: parameter.grad for name, parameter in module.named_parameters()})
    return out


def _sinkhorn(logits: torch.Tensor, iterations: int, eps: float) -> torch.Tensor:
    M = logits.softmax(dim=-1) + eps
    M = M / (M.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


def _fp64(params: dict, inputs: dict) -> dict:
    """The connection and the stand-in branch in FP64, and their gradients by autograd."""
    T, config = S * B, _config()
    eps = 1e-6  # Megatron's coefficient and Sinkhorn epsilon
    with torch.enable_grad():
        X = inputs["hidden_states"].to(F64).reshape(T, N, H).clone().requires_grad_(True)
        p = {name: value.to(F64).requires_grad_(True) for name, value in params.items()}
        u = X.reshape(T, N * H)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
        alpha = torch.cat([p["alpha_pre"].expand(N), p["alpha_post"].expand(N), p["alpha_res"].expand(N * N)])
        logits = (u * r) @ p["mapping_proj.weight"].T * alpha + p["bias"]
        pre = torch.sigmoid(logits[:, :N]) + eps
        post = 2 * torch.sigmoid(logits[:, N : 2 * N])
        C = _sinkhorn(logits[:, 2 * N :].reshape(T, N, N), config.mhc_sinkhorn_iterations, eps)
        U = (pre.unsqueeze(-1) * X).sum(dim=1)
        x = U * inputs["w"].to(F64).reshape(T, H)
        Y = post.unsqueeze(-1) * x.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, X)
        grads = torch.autograd.grad(Y, [X, *p.values()], grad_outputs=inputs["dY"].to(F64).reshape(T, N, H))
    out = {"U": U.detach().view(S, B, H), "Y": Y.detach().view(S, B, N * H), "dX": grads[0].view(S, B, N * H)}
    out.update(zip(p, grads[1:], strict=True))
    return out


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.to(F64) - reference).norm() / reference.norm())


def _assert_bitwise(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, label
    bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    assert torch.equal(a.contiguous().view(bits), b.contiguous().view(bits)), label


@pytest.mark.parametrize("name", ["det", "fast"])
def test_cake_and_megatron_are_within_bf16_coefficient_tolerance_of_fp64(params, inputs, variant, name):
    variant(name)
    reference = _fp64(params, inputs)
    megatron = _run(_built(HyperConnectionModule, params), inputs)
    cake = _run(_built(CakeHyperConnection, params), inputs)

    errors = {
        key: (_relative_l2(cake[key], reference[key]), _relative_l2(megatron[key], reference[key]))
        for key in TOLERANCES
    }
    for key, tolerance in TOLERANCES.items():
        assert max(errors[key]) <= tolerance, f"{key}: relative L2 (cake, megatron) {errors[key]} > {tolerance}"


def test_det_repeats_bitwise_and_under_tensor_parallel_checkpoint(params, inputs):
    module = _built(CakeHyperConnection, params)
    first, second = _run(module, inputs), _run(module, inputs)
    checkpointed = _run(module, inputs, checkpointed=True)

    for key, value in first.items():
        _assert_bitwise(second[key], value, f"run to run {key}")
        if key != "U":
            _assert_bitwise(checkpointed[key], value, f"checkpointed {key}")


def test_fast_forward_and_input_gradient_are_the_det_bits(params, inputs, variant):
    module = _built(CakeHyperConnection, params)
    det = _run(module, inputs)
    variant("fast")
    fast = _run(module, inputs)

    for key in ("U", "Y", "dX", "alpha_pre", "alpha_post", "alpha_res", "bias"):
        _assert_bitwise(fast[key], det[key], key)
    assert _relative_l2(fast["mapping_proj.weight"], det["mapping_proj.weight"].to(F64)) <= 1e-5


def test_the_cake_head_is_megatrons_stream_mean_bit_for_bit(inputs):
    head = CakeMeanOutputContraction(_config())
    cake_input = inputs["hidden_states"].clone().requires_grad_(True)
    megatron_input = inputs["hidden_states"].clone().requires_grad_(True)
    dY = inputs["w"]

    Y = head(cake_input)
    reference = HyperConnectionModule.output_contract(megatron_input, N)
    _assert_bitwise(Y, reference, "head Y")
    Y.backward(dY)
    reference.backward(dY)
    _assert_bitwise(cake_input.grad, megatron_input.grad, "head dX")


def _block_spec() -> SimpleNamespace:
    layer = SimpleNamespace(
        submodules=SimpleNamespace(
            self_attention_hyper_connection=HyperConnectionModule, mlp_hyper_connection=HyperConnectionModule
        )
    )
    return SimpleNamespace(layer_specs=[layer], hc_head_contraction=None)


def test_apply_mhc_backend_checks_the_device_and_configures_the_kernels():
    args = SimpleNamespace(mhc_backend="cake", miles_ext_kernels_args={"cake_mhc_variant": "fast"})
    spec = apply_mhc_backend(_block_spec(), args, _config(), mean_head=True)

    submodules = spec.layer_specs[0].submodules
    assert submodules.self_attention_hyper_connection is submodules.mlp_hyper_connection is CakeHyperConnection
    assert spec.hc_head_contraction.module is CakeMeanOutputContraction
    assert ops.VARIANT == "fast"
    small = SimpleNamespace(num_residual_streams=N, hidden_size=2048, use_fused_mhc=False)
    with pytest.raises(ValueError, match="hidden size 2048"):
        apply_mhc_backend(_block_spec(), args, small)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
