"""``--moe-grouped-gemm-backend cake`` without a device: the model-config conversion of the torchtitan backend and the
grouped matmul of ``CakeGroupedLinear``.

The kernels are a stand-in here. Their numerics and argument rules are tested in the miles-ext-kernels submodule
(``3rdparty/miles-ext-kernels/tests/cake/moe_grouped_gemm``), and ``CakeGroupedLinear`` against
``torch._grouped_mm`` on Blackwell in ``tests/fast-gpu/test_cake_grouped_linear.py``.
"""

import dataclasses
import sys
import types

import pytest
import torch

pytest.importorskip("torchtitan")

from tests.fast.backends.test_torchtitan import _config_args  # noqa: E402
from torchtitan.config.parallelism import ParallelismConfig  # noqa: E402
from torchtitan.models.common.linear import GroupedLinear  # noqa: E402
from torchtitan.models.common.moe import RoutedExperts  # noqa: E402

from miles.backends.torchtitan_utils import cake_grouped_linear  # noqa: E402
from miles.backends.torchtitan_utils.cake_grouped_linear import CakeGroupedLinear  # noqa: E402
from miles.backends.torchtitan_utils.config import build_model_config  # noqa: E402

GLM_FLASH = dict(titan_model_name="glm4_moe_lite", titan_model_flavor="30B-A3B")
DENSE = dict(titan_model_name="qwen3", titan_model_flavor="0.6B")


@pytest.fixture
def kernels(monkeypatch):
    """A stand-in op module ``miles_ext_kernels.cake.moe_grouped_gemm``, with ``miles_ext_kernels.configure`` and
    ``miles_ext_kernels.cake.prebuild``, recording their calls."""
    calls = []
    op = types.ModuleType("miles_ext_kernels.cake.moe_grouped_gemm")

    def check_grouped_dims(n, k, *, num_groups=None):
        calls.append(("check_grouped_dims", n, k, num_groups))
        if n % 256 or k % 256:
            raise ValueError(f"N = {n} and K = {k} must be multiples of 256")

    op.check_grouped_dims = check_grouped_dims
    op.grouped_mm = lambda x, w, offs: calls.append(("grouped_mm", x, w, offs)) or x.new_zeros(x.shape[0], w.shape[1])
    package = types.ModuleType("miles_ext_kernels")
    package.configure = lambda values: calls.append(("configure", values))
    cake = types.ModuleType("miles_ext_kernels.cake")
    cake.prebuild = lambda name: calls.append(("prebuild", name)) or []
    package.cake, cake.moe_grouped_gemm = cake, op
    for module in (package, cake, op):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.fixture
def no_kernels(monkeypatch):
    for name in ("miles_ext_kernels", "miles_ext_kernels.cake", "miles_ext_kernels.cake.moe_grouped_gemm"):
        monkeypatch.setitem(sys.modules, name, None)  # an import of it raises ModuleNotFoundError


def _model_config(backend: str, model=GLM_FLASH, ext_args=None):
    args = _config_args(**model)
    args.moe_grouped_gemm_backend = backend
    args.miles_ext_kernels_args = ext_args
    return build_model_config(args, parallelism=ParallelismConfig())


def _grouped_linears(model_config):
    return [(fqn, config, parent) for fqn, config, parent, _ in model_config.traverse(GroupedLinear.Config)]


def test_the_torch_backend_keeps_torchtitan_grouped_linears(no_kernels):  # the kernels are never imported

    linears = _grouped_linears(_model_config("torch"))

    assert linears and {type(config) for _, config, _ in linears} == {GroupedLinear.Config}


def _init_fields(config) -> dict:
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config) if field.init}


def test_build_model_config_converts_under_the_cake_backend(kernels):
    linears = _grouped_linears(_model_config("cake", ext_args={"cake_moe_grouped_gemm_variant": "det"}))

    assert linears and {type(config) for _, config, _ in linears} == {CakeGroupedLinear.Config}
    assert {fqn.rsplit(".", 1)[-1] for fqn, _, _ in linears} == {"w13", "w2"}
    assert kernels[0] == ("configure", {"cake_moe_grouped_gemm_variant": "det"})


def test_every_routed_expert_projection_keeps_its_fields(kernels):
    model_config = _model_config("torch")
    before = _grouped_linears(model_config)

    after = _grouped_linears(
        cake_grouped_linear.convert_model_config(
            model_config, backend="cake", miles_ext_kernels_args=None, expert_parallel_degree=1
        )
    )

    assert [fqn for fqn, _, _ in after] == [fqn for fqn, _, _ in before]
    for (_, old, _), (fqn, new, parent) in zip(before, after, strict=True):
        assert type(new) is CakeGroupedLinear.Config and isinstance(parent, RoutedExperts.Config), fqn
        old_fields, new_fields = _init_fields(old), _init_fields(new)
        assert new_fields.keys() == old_fields.keys(), fqn
        assert all(new_fields[name] is value for name, value in old_fields.items()), fqn


def test_the_conversion_is_idempotent(kernels):
    model_config = _model_config("cake")

    again = cake_grouped_linear.convert_model_config(
        model_config, backend="cake", miles_ext_kernels_args=None, expert_parallel_degree=1
    )

    assert {type(config) for _, config, _ in _grouped_linears(again)} == {CakeGroupedLinear.Config}


def test_a_dense_model_is_refused(kernels):
    with pytest.raises(ValueError, match="no MoE layer"):
        _model_config("cake", model=DENSE)


def test_expert_widths_the_kernels_do_not_serve_are_refused_with_the_projection(kernels):
    model_config = _model_config("torch")
    fqn, config, _ = _grouped_linears(model_config)[0]
    config.in_features = 2000

    with pytest.raises(ValueError, match=f"cannot run {fqn}: .*multiples of 256"):
        cake_grouped_linear.convert_model_config(
            model_config, backend="cake", miles_ext_kernels_args=None, expert_parallel_degree=1
        )


@pytest.mark.parametrize("expert_parallel_degree", [1, 4])
def test_the_expert_weights_of_one_rank_are_checked(kernels, expert_parallel_degree):
    """The 32-bit offset bound of the weight gradient covers the experts one expert-parallel rank holds."""
    model_config = _model_config("torch")

    cake_grouped_linear.convert_model_config(
        model_config, backend="cake", miles_ext_kernels_args=None, expert_parallel_degree=expert_parallel_degree
    )

    checks = [call for call in kernels if call[0] == "check_grouped_dims"]
    linears = _grouped_linears(model_config)
    assert len(checks) == len(linears)
    for (_, n, k, num_groups), (fqn, config, _) in zip(checks, linears, strict=True):
        assert (n, k) == (config.out_features * config.num_linears, config.in_features), fqn
        assert num_groups == config.group_size // expert_parallel_degree, fqn


class _AdaptedGroupedLinear(GroupedLinear):
    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass


def test_a_quantized_or_adapted_grouped_linear_is_refused(kernels):
    model_config = _model_config("torch")
    parent = _grouped_linears(model_config)[0][2]
    parent.w2 = _AdaptedGroupedLinear.Config(**_init_fields(parent.w2))

    with pytest.raises(NotImplementedError, match="_AdaptedGroupedLinear"):
        cake_grouped_linear.convert_model_config(
            model_config, backend="cake", miles_ext_kernels_args=None, expert_parallel_degree=1
        )


def test_the_grouped_matmul_hands_the_kernels_bf16_weights_in_torch_grouped_mm_form(kernels):
    linear = CakeGroupedLinear.Config(group_size=4, in_features=256, out_features=512).build()
    assert isinstance(linear, CakeGroupedLinear) and linear.weight.shape == (4, 512, 256)
    x = torch.zeros(10, 256, dtype=torch.bfloat16)
    weight = torch.zeros(4, 512, 256, dtype=torch.float32)
    offs = torch.tensor([3, 3, 7, 10], dtype=torch.int32)

    out = linear._grouped_mm(input_RI=x, weight_EOI=weight, offsets_E=offs)

    (name, got_x, got_w, got_offs) = kernels[-1]
    assert name == "grouped_mm" and got_x is x and got_offs is offs
    assert got_w.dtype == torch.bfloat16 and got_w.shape == (4, 512, 256)
    assert out.shape == (10, 512)


@pytest.mark.parametrize("backend, builds", [("torch", 0), ("cake", 1)])
def test_prebuild_runs_only_for_the_cake_backend(kernels, backend, builds):
    cake_grouped_linear.prebuild(backend=backend)

    assert kernels.count(("prebuild", "moe_grouped_gemm")) == builds


def test_a_missing_submodule_names_the_flag_that_needs_it(no_kernels):
    with pytest.raises(ImportError, match="--moe-grouped-gemm-backend cake .*--use-miles-ext-kernels"):
        _model_config("cake")
