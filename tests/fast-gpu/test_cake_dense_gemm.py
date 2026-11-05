"""``--batched-projection-gemm-backend cake`` and ``--moe-router-gemm-backend cake`` on SM100a / SM103a / SM107a
against torch and Megatron, through ``miles_plugins.models.cake_batched_projection``,
``miles.backends.megatron_utils.cake_router_gemm`` and the real kernel package:

* the MLA q-absorb and value projections of GLM-5 / GLM-5.3 on the views their attention takes (a slice of a wider head
  slot, a split of ``linear_kv_up_proj.weight``) and DeepSeek-V4's grouped output projection: the output and both
  gradients within BF16 bounds of FP64, as ``torch.einsum``, and bit-identical reruns;
* the router gate of a Megatron router rebound by ``install_cake_router_gemm``: FP32 logits and the BF16 gradients
  within FP32 / BF16 bounds of FP64, as Megatron's ``Router.gating``, the same top-k experts outside near ties, and
  bit-identical reruns.

    python tests/fast-gpu/test_cake_dense_gemm.py

The kernels' own suites (more shapes, the prepared launches, the prebuild hooks) run in
``3rdparty/miles-ext-kernels/tests/cake/dense_gemm`` through ``tests/fast-gpu/kernels/test_miles_ext_kernels.py``.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the Cake dense GEMMs run on NVIDIA SM100a / SM103a / SM107a",
)

# a missing submodule fails the collection: on the Blackwell runners a skip would hide the coverage
import miles_ext_kernels.cake.dense_gemm  # noqa: E402, F401
from megatron.core.transformer.moe.router import Router  # noqa: E402

from miles.backends.megatron_utils import cake_router_gemm  # noqa: E402
from miles_plugins.models.cake_batched_projection import (  # noqa: E402
    grouped_matmul,
    project_heads,
    validate_batched_projection_config,
)

SEED = 20260930
BF16_ATOL = BF16_RTOL = 1e-2
TOKENS = 1001
# (heads per tensor-parallel rank, the q / k head dim the absorb reads, v head dim, kv_lora_rank)
MLA = {"glm5": (16, 192, 256, 512), "glm53": (16, 256, 256, 512)}
# DeepSeek-V4's output groups per rank, their input width and o_lora_rank
DSV4 = (8, 4096, 1024)
TOP_K = 8
# Top-k agreement is checked on the rows whose k-th and (k+1)-th FP64 logits are further apart than this (relative to
# the largest logit): closer ties may order either way under any FP32 accumulation.
TOP_K_TIE_MARGIN = 1e-4
# The share of rows that must be clear of such a tie for the check to cover the routing. It depends on the inputs only
# (the FP64 logits): at this seed 989 of the 1001 rows (98.8 %) of the 384-expert case are clear.
TOP_K_MIN_CLEAR_ROWS = 0.95


@pytest.fixture(autouse=True)
def _fp64_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _randn(*shape, scale=1.0, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)


def _grads(fn, leaves, grad, dtype=None):
    """``fn(*leaves)`` and the gradients of its leaves, with the leaves (and the output gradient) in ``dtype``."""
    leaves = [leaf.detach().to(dtype or leaf.dtype).clone().requires_grad_(True) for leaf in leaves]
    out = fn(*leaves)
    out.backward(grad.to(out.dtype))
    return (out.detach(), *(leaf.grad for leaf in leaves))


def _assert_bf16_close(name, got, want):
    assert got.dtype == torch.bfloat16, name
    torch.testing.assert_close(got.double(), want, atol=BF16_ATOL, rtol=BF16_RTOL, msg=lambda m: f"{name}: {m}")


def _check(fn, leaves, grad):
    """``fn`` under cake: within BF16 bounds of its FP64 evaluation, as under torch, and bitwise rerun."""
    cake, again = (_grads(lambda *a: fn(*a, "cake"), leaves, grad) for _ in range(2))
    torch_ = _grads(lambda *a: fn(*a, "torch"), leaves, grad)
    reference = _grads(lambda *a: fn(*a, "torch"), leaves, grad, dtype=torch.float64)
    for index, (got, rerun, eager, want) in enumerate(zip(cake, again, torch_, reference, strict=True)):
        assert torch.equal(got, rerun), f"result {index} differs between two identical calls"
        _assert_bf16_close(f"cake result {index}", got, want)
        _assert_bf16_close(f"torch result {index}", eager, want)


@pytest.mark.parametrize("model", list(MLA))
def test_the_mla_projections_on_the_attention_views(model):
    heads, qk_dim, v_dim, rank = MLA[model]
    # linear_kv_up_proj.weight [H * (qk + v), rank], split per head as the attention does
    kv_up = _randn(heads * (qk_dim + v_dim), rank, scale=rank**-0.5, seed=SEED)
    q = _randn(TOKENS, heads, qk_dim + 64, seed=SEED + 1)  # the no-pe part of each head slot, then the rope part

    def absorb(q, kv_up, backend):
        w_kc, _ = kv_up.unflatten(0, (-1, qk_dim + v_dim)).split([qk_dim, v_dim], dim=1)
        return project_heads(q[..., :qk_dim], w_kc, weight_layout="in_out", backend=backend)

    def value(o, kv_up, backend):
        _, w_vc = kv_up.unflatten(0, (-1, qk_dim + v_dim)).split([qk_dim, v_dim], dim=1)
        return project_heads(o, w_vc, weight_layout="out_in", backend=backend)

    _check(absorb, (q, kv_up), grad=_randn(TOKENS, heads, rank, seed=SEED + 2))
    _check(
        value, (_randn(TOKENS, heads, rank, seed=SEED + 3), kv_up), grad=_randn(TOKENS, heads, v_dim, seed=SEED + 4)
    )


def test_the_dsv4_grouped_output_projection():
    groups, width, rank = DSV4
    o = _randn(1, TOKENS, groups, width, seed=SEED + 5)
    weight = _randn(groups * rank, width, scale=width**-0.5, seed=SEED + 6)

    _check(
        lambda o, w, backend: grouped_matmul(o, w, backend=backend),
        (o, weight),
        grad=_randn(1, TOKENS, groups, rank, seed=SEED + 7),
    )


def test_the_layer_check_takes_the_mla_shapes_and_refuses_others():
    """The check where the layers are built, which also builds the projection programs of this GPU."""
    config = SimpleNamespace(cuda_graph_impl="none", params_dtype=torch.bfloat16)
    for num_heads, d_in, d_out, weight_layout in (
        (16, 192, 512, "in_out"),
        (16, 512, 256, "out_in"),
        (16, 256, 512, "in_out"),
        (8, 4096, 1024, "out_in"),
    ):
        validate_batched_projection_config(
            config, num_heads=num_heads, d_in=d_in, d_out=d_out, weight_layout=weight_layout, backend="cake"
        )
    with pytest.raises(ValueError, match="multiple of 8"):
        validate_batched_projection_config(
            config, num_heads=16, d_in=196, d_out=512, weight_layout="in_out", backend="cake"
        )


class _Router(Router):
    """A Megatron router without process groups (Router.__init__ is skipped): Megatron's own gating."""

    def __init__(self, config, weight):
        torch.nn.Module.__init__(self)
        self.config = config
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = None

    def routing(self, logits):
        return logits

    def forward(self, input):
        return self.gating(input)


def _router_config(experts, hidden, use_torch_mm):
    return SimpleNamespace(
        moe_router_dtype="fp32",
        add_bias_linear=False,
        moe_router_use_torch_mm=use_torch_mm,
        cuda_graph_impl="none",
        params_dtype=torch.bfloat16,
        num_moe_experts=experts,
        hidden_size=hidden,
    )


def _router_step(router, x, grad):
    router.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    logits = router(x)
    logits.backward(grad)
    return logits.detach(), x.grad, router.weight.grad


# use_torch_mm=True is what every miles run sets (set_default_megatron_args); it picks Megatron's reference GEMM only
@pytest.mark.parametrize("use_torch_mm", [False, True])
@pytest.mark.parametrize("experts, hidden", [(256, 6144), (384, 7168), (160, 5120)])
def test_the_cake_router_gate_as_megatron_router_gating(experts, hidden, use_torch_mm):
    config = _router_config(experts, hidden, use_torch_mm)
    weight = _randn(experts, hidden, scale=0.02, seed=SEED + experts)
    x = _randn(TOKENS, 1, hidden, seed=SEED + hidden)
    grad = (
        torch.randn(TOKENS, 1, experts, device="cuda", generator=torch.Generator(device="cuda").manual_seed(SEED))
        * 1e-3
    )
    megatron, cake = _Router(config, weight), _Router(config, weight)
    model = torch.nn.ModuleDict({"cake": cake})
    assert cake_router_gemm.install_cake_router_gemm(model) == 1

    want = _router_step(megatron, x, grad)
    got, again = _router_step(cake, x, grad), _router_step(cake, x, grad)

    for name, a, b in zip(("logits", "input gradient", "weight gradient"), got, again, strict=True):
        assert torch.equal(a, b), f"{name} differs between two identical calls"
    x64, w64, g64 = x.double().reshape(-1, hidden), weight.double(), grad.double().reshape(-1, experts)
    expected = (x64 @ w64.T, g64 @ w64, g64.T @ x64)
    logits = got[0].reshape(-1, experts)
    assert logits.dtype == torch.float32
    assert float((logits.double() - expected[0]).norm() / expected[0].norm()) <= 5e-5
    for name, index in (("input gradient", 1), ("weight gradient", 2)):
        for side, values in (("cake", got), ("megatron", want)):
            out = values[index].reshape(expected[index].shape)
            assert out.dtype == torch.bfloat16, (side, name)
            rel = float((out.double() - expected[index]).norm() / expected[index].norm())
            assert rel <= 2**-8, f"{side} {name}: relative error {rel:.2e}"

    ranked = torch.topk(expected[0], TOP_K + 1, dim=1).values
    clear = (ranked[:, TOP_K - 1] - ranked[:, TOP_K]) > TOP_K_TIE_MARGIN * float(expected[0].abs().max())
    assert int(clear.sum()) >= TOP_K_MIN_CLEAR_ROWS * TOKENS, "too many near ties to check the routing"
    top = {
        side: torch.topk(values[0].reshape(-1, experts), TOP_K, dim=1).indices.sort(dim=1).values[clear]
        for side, values in (("cake", got), ("megatron", want))
    }
    assert torch.equal(top["cake"], top["megatron"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
