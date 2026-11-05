"""``CakeGroupedLinear`` (``--moe-grouped-gemm-backend cake``) on Blackwell against torchtitan's ``GroupedLinear``
(``torch._grouped_mm``): the gate / up and down projections of the routed experts of a GLM-4.7-Flash layer on one
expert-parallel rank (8 of its 64 experts, hidden 2048, expert width 1536), over ragged groups with an empty and a
one-row one. The output and both gradients within BF16 bounds of a per-group FP32 reference, as torch's; ``det``
bitwise run to run; ``fast`` with ``det``'s output and input-gradient bits.

The kernels' own suites (more layer shapes, the unordered 512-wide k tail, ``fast`` running ``det`` under torch's
deterministic algorithms, the prebuild hook) run in ``3rdparty/miles-ext-kernels/tests/cake/moe_grouped_gemm`` through
``tests/fast-gpu/kernels/test_miles_ext_kernels.py``.

    python tests/fast-gpu/test_cake_grouped_linear.py
"""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=1200,
    suite="stage-c-4-gpu-b200",
    labels=["precision", "torchtitan"],
    hardware=["blackwell"],
    num_gpus=1,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the Cake MoE grouped GEMM runs on NVIDIA SM100a / SM103a / SM107a",
)

# a missing torchtitan or submodule fails the collection: on the Blackwell runners a skip would hide the coverage
import miles_ext_kernels  # noqa: E402
from miles_ext_kernels import cake  # noqa: E402
from torchtitan.models.common.linear import GroupedLinear  # noqa: E402

from miles.backends.torchtitan_utils.cake_grouped_linear import CakeGroupedLinear  # noqa: E402

SEED = 20260930
ATOL = RTOL = 1e-2
HIDDEN, EXPERT_WIDTH, LOCAL_EXPERTS = 2048, 1536, 8
SIZES = [300, 0, 1, 700, 4099, 33, 511, 257]
PROJECTIONS = {
    "w13": dict(in_features=HIDDEN, out_features=EXPERT_WIDTH, num_linears=2),
    "w2": dict(in_features=EXPERT_WIDTH, out_features=HIDDEN, num_linears=1),
}


@pytest.fixture(scope="module", autouse=True)
def _programs():
    cake.prebuild("moe_grouped_gemm")  # both variants


@pytest.fixture(autouse=True)
def _fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _linears(proj: str):
    """torchtitan's and the Cake grouped linear of ``proj``, on the device with one BF16 weight."""
    fields = dict(group_size=LOCAL_EXPERTS, **PROJECTIONS[proj])
    torch_linear, cake_linear = GroupedLinear.Config(**fields).build(), CakeGroupedLinear.Config(**fields).build()
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    weight = torch.randn(torch_linear.weight.shape, generator=gen, device="cuda") * fields["in_features"] ** -0.5
    for linear in (torch_linear, cake_linear):
        linear.weight = torch.nn.Parameter(weight.to(torch.bfloat16))
    return torch_linear, cake_linear


def _inputs(proj: str):
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    fields = PROJECTIONS[proj]
    rows = sum(SIZES)
    x = torch.randn(rows, fields["in_features"], generator=gen, device="cuda").to(torch.bfloat16)
    out_shape = (rows, fields["out_features"]) if fields["num_linears"] == 1 else (rows, 2, fields["out_features"])
    grad = torch.randn(out_shape, generator=gen, device="cuda").to(torch.bfloat16)
    offs = torch.tensor(SIZES, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return x, grad, offs


def _step(linear, x, grad, offs):
    linear.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    out = linear(x, offs)
    out.backward(grad)
    torch.cuda.synchronize()
    return out.detach(), x.grad, linear.weight.grad


def _reference(weight, x, grad):
    """Per-group FP32 products of the BF16 operands, ``weight`` flattened to [E, O, I]: ``(y, dx, dw)``."""
    w = weight.flatten(1, -2).float()
    g = grad.reshape(grad.shape[0], -1).float()
    y, dx, dw = torch.zeros_like(g), torch.zeros(x.shape, device="cuda"), torch.zeros_like(w)
    start = 0
    for e, size in enumerate(SIZES):
        rows = slice(start, start + size)
        if size:
            y[rows] = x[rows].float() @ w[e].T
            dx[rows] = g[rows] @ w[e]
            dw[e] = g[rows].T @ x[rows].float()
        start += size
    return y.view(grad.shape), dx, dw.view(weight.shape)


def _configure(variant: str) -> None:
    miles_ext_kernels.configure(cake_moe_grouped_gemm_variant=variant)


@pytest.mark.parametrize("proj", list(PROJECTIONS))
def test_cake_grouped_linear_as_torch_grouped_mm(proj):
    torch_linear, cake_linear = _linears(proj)
    x, grad, offs = _inputs(proj)
    reference = _reference(cake_linear.weight, x, grad)
    _configure("det")

    torch_out = _step(torch_linear, x, grad, offs)
    det, det_again = _step(cake_linear, x, grad, offs), _step(cake_linear, x, grad, offs)

    for name, got, again, eager, want in zip(("y", "dx", "dw"), det, det_again, torch_out, reference, strict=True):
        assert torch.equal(got, again), f"det {name} differs between two identical calls"
        for side, value in (("cake", got), ("torch", eager)):
            assert value.dtype == torch.bfloat16 and value.shape == want.shape, (side, name)
            label = f"{side} {name}"
            torch.testing.assert_close(
                value.float(), want, atol=ATOL, rtol=RTOL, msg=lambda m, label=label: f"{label}: {m}"
            )
    for e, size in enumerate(SIZES):
        if not size:
            assert not det[2][e].any(), f"the weight gradient of empty expert {e} must be exactly zero"


@pytest.mark.parametrize("proj", list(PROJECTIONS))
def test_fast_gives_the_det_output_and_input_gradient(proj):
    _, cake_linear = _linears(proj)
    x, grad, offs = _inputs(proj)
    reference = _reference(cake_linear.weight, x, grad)

    _configure("det")
    det = _step(cake_linear, x, grad, offs)
    _configure("fast")
    fast = _step(cake_linear, x, grad, offs)

    assert torch.equal(fast[0], det[0]) and torch.equal(fast[1], det[1])
    torch.testing.assert_close(fast[2].float(), reference[2], atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
