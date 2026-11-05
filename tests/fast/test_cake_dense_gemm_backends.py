"""``--batched-projection-gemm-backend`` and ``--moe-router-gemm-backend`` without a device: the call contract, the
startup checks and the router installation of ``miles_plugins.models.cake_batched_projection`` and
``miles.backends.megatron_utils.cake_router_gemm``.

The kernels are a stand-in here. Their numerics and argument rules are tested in the miles-ext-kernels submodule
(``3rdparty/miles-ext-kernels/tests/cake/dense_gemm``), and these modules against torch / Megatron on Blackwell in
``tests/fast-gpu/test_cake_dense_gemm.py``.
"""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.router import Router

from miles.backends.megatron_utils import cake_router_gemm, model_provider
from miles_plugins.models import cake_batched_projection


@pytest.fixture
def kernels(monkeypatch):
    """A stand-in op module ``miles_ext_kernels.cake.dense_gemm`` computing with torch on the host, and
    ``miles_ext_kernels.cake.prebuild``, recording their calls."""
    calls = []
    dense_gemm = types.ModuleType("miles_ext_kernels.cake.dense_gemm")

    def batched_projection(x, weight, *, weight_layout):
        calls.append(("batched_projection", weight_layout, tuple(x.shape), tuple(weight.shape)))
        equation = {"in_out": "thd,hdm->thm", "out_in": "thm,hdm->thd"}[weight_layout]
        return torch.einsum(equation, x, weight)

    def check_projection_dims(d_in, d_out):
        calls.append(("check_projection_dims", d_in, d_out))
        if d_in % 8 or d_out % 8:
            raise ValueError("not a multiple of 8")

    def router_gating(x, weight):
        calls.append(("router_gating", tuple(x.shape), tuple(weight.shape)))
        return x.float() @ weight.float().t()

    dense_gemm.batched_projection = batched_projection
    dense_gemm.check_projection_dims = check_projection_dims
    dense_gemm.router_gating = router_gating
    package = types.ModuleType("miles_ext_kernels")
    cake = types.ModuleType("miles_ext_kernels.cake")
    cake.prebuild = lambda op, **model_params: calls.append(("prebuild", op, model_params)) or []
    package.cake, cake.dense_gemm = cake, dense_gemm
    for module in (package, cake, dense_gemm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.fixture
def no_kernels(monkeypatch):
    for name in ("miles_ext_kernels", "miles_ext_kernels.cake", "miles_ext_kernels.cake.dense_gemm"):
        monkeypatch.setitem(sys.modules, name, None)  # an import of it raises ModuleNotFoundError


def _config(**overrides):
    values = dict(
        cuda_graph_impl="none",
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        add_bias_linear=False,
        moe_router_use_torch_mm=False,
        num_moe_experts=256,
        hidden_size=6144,
    )
    return SimpleNamespace(**{**values, **overrides})


# --- batched projection ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["torch", "cake"])
@pytest.mark.parametrize("weight_layout, d_in, d_out", [("in_out", 192, 512), ("out_in", 512, 256)])
def test_project_heads_is_the_einsum_of_its_layout(kernels, backend, weight_layout, d_in, d_out):
    x = torch.randn(5, 4, d_in, dtype=torch.float64)
    weight = torch.randn(4, *((d_in, d_out) if weight_layout == "in_out" else (d_out, d_in)), dtype=torch.float64)
    equation = {"in_out": "thd,hdm->thm", "out_in": "thm,hdm->thd"}[weight_layout]

    out = cake_batched_projection.project_heads(x, weight, weight_layout=weight_layout, backend=backend)

    torch.testing.assert_close(out, torch.einsum(equation, x, weight))
    launched = [call for call in kernels if call[0] == "batched_projection"]
    assert launched == (
        [("batched_projection", weight_layout, (5, 4, d_in), tuple(weight.shape))] if backend == "cake" else []
    )


@pytest.mark.parametrize("backend", ["torch", "cake"])
def test_grouped_matmul_is_the_grouped_einsum(kernels, backend):
    x = torch.randn(2, 3, 4, 16, dtype=torch.float64)  # [b, s, g, d]
    weight = torch.randn(4 * 8, 16, dtype=torch.float64)  # [g * r, d]

    out = cake_batched_projection.grouped_matmul(x, weight, backend=backend)

    assert out.shape == (2, 3, 4, 8)
    torch.testing.assert_close(out, torch.einsum("bsgd,grd->bsgr", x, weight.view(4, 8, 16)))
    if backend == "cake":  # the [b * s, g, d] tokens times the [g, r, d] (out_in) weight blocks
        assert ("batched_projection", "out_in", (6, 4, 16), (4, 8, 16)) in kernels


def test_an_unknown_backend_or_layout_is_refused(kernels):
    x, weight = torch.zeros(1, 1, 8), torch.zeros(1, 8, 8)
    with pytest.raises(ValueError, match="backend must be one of"):
        cake_batched_projection.project_heads(x, weight, weight_layout="in_out", backend="cublas")
    with pytest.raises(ValueError, match="weight_layout must be one of"):
        cake_batched_projection.project_heads(x, weight, weight_layout="io", backend="torch")


def test_the_torch_backend_checks_nothing(no_kernels):
    config = _config(cuda_graph_impl="local", params_dtype=torch.float32)

    cake_batched_projection.validate_batched_projection_config(
        config, num_heads=4, d_in=3, d_out=5, weight_layout="in_out", backend="torch"
    )


def test_the_cake_backend_accepts_the_mla_projections_and_builds_their_programs(kernels):
    # GLM-5's q-absorb and value projections, DeepSeek-V4's grouped output projection
    for num_heads, d_in, d_out, weight_layout in (
        (32, 192, 512, "in_out"),
        (32, 512, 256, "out_in"),
        (16, 4096, 1024, "out_in"),
    ):
        cake_batched_projection.validate_batched_projection_config(
            _config(), num_heads=num_heads, d_in=d_in, d_out=d_out, weight_layout=weight_layout, backend="cake"
        )

    assert kernels[-2:] == [
        ("check_projection_dims", 4096, 1024),
        ("prebuild", "dense_gemm", {"projections": [(16, 4096, 1024, "out_in")]}),
    ]


@pytest.mark.parametrize(
    "overrides, dims, match",
    [
        (dict(cuda_graph_impl="transformer_engine"), (192, 512), "CUDA graphs"),
        (dict(params_dtype=torch.float16), (192, 512), "BF16 weights"),
        ({}, (192, 500), "multiple of 8"),
    ],
)
def test_the_cake_backend_refuses_what_the_kernel_does_not_run(kernels, overrides, dims, match):
    with pytest.raises(ValueError, match=match):
        cake_batched_projection.validate_batched_projection_config(
            _config(**overrides), num_heads=32, d_in=dims[0], d_out=dims[1], weight_layout="in_out", backend="cake"
        )
    assert not [call for call in kernels if call[0] == "prebuild"]


def test_a_missing_submodule_names_the_flag_that_needs_it(no_kernels):
    with pytest.raises(ImportError, match="--batched-projection-gemm-backend cake .*--use-miles-ext-kernels"):
        cake_batched_projection.project_heads(
            torch.zeros(1, 1, 8), torch.zeros(1, 8, 8), weight_layout="in_out", backend="cake"
        )


# --- router gate GEMM -----------------------------------------------------------------------------------------------


class _Router(Router):
    """A Megatron router without process groups, its weight on the meta device (Router.__init__ is skipped)."""

    def __init__(self, config, device="meta"):
        torch.nn.Module.__init__(self)
        self.config = config
        self.weight = torch.nn.Parameter(torch.empty(config.num_moe_experts, config.hidden_size, device=device))
        self.bias = None

    def routing(self, logits):
        return logits

    def forward(self, input):
        return self.gating(input)


class _OwnGatingRouter(_Router):
    def gating(self, input):
        return input


def _model(*routers):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(torch.nn.ModuleDict({"router": router}) for router in routers)
    return model


def test_the_router_config_the_kernels_reproduce_is_accepted():
    cake_router_gemm.validate_router_config(_config())


def test_the_router_config_of_every_miles_run_is_accepted():
    # miles sets --moe-router-use-torch-mm for every Megatron run; it only picks the GEMM inside the Router.gating
    # that the Cake kernels replace
    cake_router_gemm.validate_router_config(_config(moe_router_use_torch_mm=True))


def test_every_router_setting_the_kernels_do_not_reproduce_is_listed():
    config = _config(
        moe_router_dtype=None,
        add_bias_linear=True,
        cuda_graph_impl="local",
        params_dtype=torch.float16,
        num_moe_experts=60,
    )

    with pytest.raises(ValueError) as info:
        cake_router_gemm.validate_router_config(config)

    message = str(info.value)
    for expected in ("--moe-router-dtype fp32", "--disable-bias-linear", "CUDA graphs"):
        assert expected in message
    assert "BF16 parameters" in message and "multiples of 8" in message


def test_install_rebinds_the_gating_of_every_router_and_builds_its_programs(kernels):
    config = _config(num_moe_experts=16, hidden_size=32)
    routers = [_Router(config), _Router(config)]

    assert cake_router_gemm.install_cake_router_gemm(_model(*routers)) == 2

    # one build for the routers' one shape, before any gating runs
    assert kernels == [("prebuild", "dense_gemm", {"router": [(16, 32)]})]

    for router in routers:
        assert router.gating.__func__ is cake_router_gemm._cake_gating and router.gating.__self__ is router
    logits = routers[1](torch.empty(3, 1, 32, device="meta"))  # the router's forward calls self.gating
    assert logits.shape == (3, 1, 16)
    assert kernels[-1] == ("router_gating", (3, 1, 32), (16, 32))


def test_a_model_without_routers_is_left_alone(no_kernels):
    assert cake_router_gemm.install_cake_router_gemm(_model()) == 0


def test_a_router_with_its_own_gating_is_refused(kernels):
    config = _config(num_moe_experts=16, hidden_size=32)
    with pytest.raises(NotImplementedError, match="_OwnGatingRouter"):
        cake_router_gemm.install_cake_router_gemm(_model(_Router(config), _OwnGatingRouter(config)))


def test_install_checks_every_router_config_before_rebinding_any(kernels):
    good, bad = _Router(_config(num_moe_experts=16, hidden_size=32)), _Router(_config(add_bias_linear=True))
    with pytest.raises(ValueError, match="--disable-bias-linear"):
        cake_router_gemm.install_cake_router_gemm(_model(good, bad))

    assert "gating" not in vars(good)
    assert kernels == []


def test_a_missing_submodule_fails_at_install(no_kernels):
    with pytest.raises(ImportError, match="--moe-router-gemm-backend cake .*--use-miles-ext-kernels"):
        cake_router_gemm.install_cake_router_gemm(_model(_Router(_config(num_moe_experts=16, hidden_size=32))))


@pytest.mark.parametrize("backend, installs", [("megatron", 0), ("cake", 1)])
def test_the_model_provider_installs_the_cake_gating_only_for_the_cake_backend(monkeypatch, backend, installs):
    installed = []
    monkeypatch.setattr(model_provider, "install_cake_router_gemm", installed.append)
    model = torch.nn.Module()

    model_provider._maybe_install_cake_router_gemm(SimpleNamespace(moe_router_gemm_backend=backend), model)

    assert installed == [model] * installs


# the per-head projection calls of each miles DSA plugin layer file: q-absorb and value (GLM), grouped output (DSv4)
_PROJECTION_SITES = {
    "miles_plugins/models/glm5/glm5.py": 2,
    "miles_plugins/models/glm5_next/dsa.py": 2,
    "miles_plugins/models/deepseek_v4/deepseek_v4.py": 1,
    "miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py": 1,
}
_PROJECTION_EQUATIONS = ("'thd,hdm->thm'", "'thm,hdm->thd'", "'bsgd,grd->bsgr'")


@pytest.mark.parametrize("relative", _PROJECTION_SITES)
def test_the_dsa_plugin_projections_take_the_backend(relative):
    """The attention layers are GPU-only to build, so the wiring is read from the source: every per-head projection
    passes the layer's ``backend``, and no projection einsum is left."""
    tree = ast.parse((Path(__file__).resolve().parents[2] / relative).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    projections = [
        call
        for call in calls
        if ast.unparse(call.func) in ("project_heads", "grouped_matmul", "GroupedMatmul")
        and "backend" in {keyword.arg for keyword in call.keywords}
    ]
    einsums = [
        call
        for call in calls
        if ast.unparse(call.func) == "torch.einsum"
        and call.args
        and ast.unparse(call.args[0]) in _PROJECTION_EQUATIONS
    ]

    assert len(projections) == _PROJECTION_SITES[relative]
    assert not einsums
