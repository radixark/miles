"""``--mhc-backend cake`` without a device: the spec swap, the module contract and the startup checks of
``miles_plugins.models.cake_hyper_connection``.

The kernel package is a stand-in here. Its numerics and its own argument rules are tested in the miles-ext-kernels
submodule (``3rdparty/miles-ext-kernels/tests/mhc``), and the module against Megatron's on Blackwell in
``tests/fast-gpu/test_cake_mhc.py``.
"""

import argparse
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.cake_hyper_connection import (
    CakeHyperConnection,
    CakeMeanOutputContraction,
    apply_mhc_backend,
    validate_mhc_backend,
)

GLM_SPEC = ["miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec"]
DSV4_SPEC = ["miles_plugins.models.deepseek_v4.deepseek_v4", "get_dsv4_spec"]
# the hidden sizes of the SM100a / SM103a kernels (SM107a: 4096)
HIDDEN_SIZES = (4096, 5120, 7168)


@pytest.fixture
def kernels(monkeypatch):
    """A stand-in ``miles_ext_kernels.mhc`` recording what the module hands it, on device 0 of capability 10.0."""
    mhc = types.ModuleType("miles_ext_kernels.mhc")
    mhc.HIDDEN_SIZES = mhc.device_hidden_sizes = HIDDEN_SIZES
    mhc.calls = []

    def is_supported(device, hidden):
        mhc.calls.append(("is_supported", device, hidden))
        return hidden in mhc.device_hidden_sizes

    def supported_hidden_sizes(device):
        return mhc.device_hidden_sizes

    def configure(args):
        mhc.calls.append(("configure", args))

    def prebuild(device, hidden, *, sinkhorn_iters, head=False):
        mhc.calls.append(("prebuild", device, hidden, sinkhorn_iters, head))

    def mhc_prepare(*args, **kwargs):
        raise AssertionError("the call should have been refused before reaching the kernels")

    mhc.is_supported, mhc.supported_hidden_sizes = is_supported, supported_hidden_sizes
    mhc.configure, mhc.prebuild = configure, prebuild
    mhc.mhc_prepare = mhc.mhc_finish = mhc_prepare
    package = types.ModuleType("miles_ext_kernels")
    package.mhc = mhc
    monkeypatch.setitem(sys.modules, "miles_ext_kernels", package)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.mhc", mhc)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    return mhc


def _block_spec(connection=HyperConnectionModule, layers: int = 2) -> SimpleNamespace:
    """The fields of a ``TransformerBlockSubmodules`` that apply_mhc_backend reads and writes."""
    return SimpleNamespace(
        layer_specs=[
            SimpleNamespace(
                submodules=SimpleNamespace(self_attention_hyper_connection=connection, mlp_hyper_connection=connection)
            )
            for _ in range(layers)
        ],
        hc_head_contraction="the model's own head",
    )


def _connections(block_spec) -> list:
    return [
        connection
        for layer in block_spec.layer_specs
        for connection in (layer.submodules.self_attention_hyper_connection, layer.submodules.mlp_hyper_connection)
    ]


def _config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"num_residual_streams": 4, "hidden_size": 4096, "use_fused_mhc": False, **overrides})


def _args(backend: str = "cake", ext_args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(mhc_backend=backend, miles_ext_kernels_args=ext_args)


def test_the_megatron_backend_leaves_the_spec_and_the_kernels_alone(kernels):
    spec = _block_spec()

    assert apply_mhc_backend(spec, _args("megatron"), _config(num_residual_streams=6), mean_head=True) is spec
    assert _connections(spec) == [HyperConnectionModule] * 4
    assert spec.hc_head_contraction == "the model's own head"
    assert kernels.calls == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_cake_swaps_both_connections_of_every_layer(kernels, wrapped):
    connection = ModuleSpec(module=HyperConnectionModule, params={"kept": 1}) if wrapped else HyperConnectionModule
    spec = _block_spec(connection)

    assert apply_mhc_backend(spec, _args(ext_args={"cake_mhc_variant": "det"}), _config()) is spec
    for swapped in _connections(spec):
        if wrapped:
            assert isinstance(swapped, ModuleSpec) and swapped.module is CakeHyperConnection
            assert swapped.params == {"kept": 1}
        else:
            assert swapped is CakeHyperConnection
    assert spec.hc_head_contraction == "the model's own head"
    assert kernels.calls == [("is_supported", 0, 4096), ("configure", argparse.Namespace(cake_mhc_variant="det"))]


def test_the_mean_head_is_swapped_only_when_asked_and_a_second_pass_changes_nothing(kernels):
    spec = _block_spec()
    apply_mhc_backend(spec, _args(), _config(), mean_head=True)
    apply_mhc_backend(spec, _args(), _config(), mean_head=True)

    assert _connections(spec) == [CakeHyperConnection] * 4
    assert isinstance(spec.hc_head_contraction, ModuleSpec)
    assert spec.hc_head_contraction.module is CakeMeanOutputContraction
    # no ext arguments (a run without --use-miles-ext-kernels values) configures the package's defaults
    assert ("configure", argparse.Namespace()) in kernels.calls


def test_layers_without_a_connection_keep_the_identity(kernels):
    spec = _block_spec(IdentityOp, layers=1)

    apply_mhc_backend(spec, _args(), _config())
    assert _connections(spec) == [IdentityOp, IdentityOp]


def test_another_connection_class_is_refused(kernels):
    class OtherHyperConnection(HyperConnectionModule):
        pass

    with pytest.raises(NotImplementedError, match="replaces Megatron's HyperConnectionModule"):
        apply_mhc_backend(_block_spec(OtherHyperConnection), _args(), _config())


@pytest.mark.parametrize(
    ("config", "device_hidden_sizes", "match"),
    [
        (_config(num_residual_streams=6), HIDDEN_SIZES, "runs 4 residual streams, not 6"),
        (_config(use_fused_mhc=True), HIDDEN_SIZES, "--use-fused-mhc"),
        (_config(hidden_size=2048), HIDDEN_SIZES, r"runs the hidden sizes \(4096, 5120, 7168\).*hidden size 2048"),
        (_config(hidden_size=5120), (4096,), r"runs the hidden sizes \(4096,\) on this device.*hidden size 5120"),
        (_config(), (), r"runs the hidden sizes \(\) on this device \(compute capability \(10, 0\)"),
    ],
)
def test_the_spec_build_refuses_what_the_kernels_do_not_run(kernels, config, device_hidden_sizes, match):
    kernels.device_hidden_sizes = device_hidden_sizes
    spec = _block_spec()

    with pytest.raises(ValueError, match=match):
        apply_mhc_backend(spec, _args(), config)
    assert _connections(spec) == [HyperConnectionModule] * 4


def _transformer_config(streams: int = 4) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        enable_hyper_connections=True,
        num_residual_streams=streams,
        hidden_dropout=0.0,
    )


def test_the_cake_module_keeps_megatrons_parameters_and_prebuilds_its_kernels(kernels):
    config = _transformer_config()
    megatron, cake = HyperConnectionModule(config, 1), CakeHyperConnection(config, 1)

    assert {name: tuple(p.shape) for name, p in cake.state_dict().items()} == {
        name: tuple(p.shape) for name, p in megatron.state_dict().items()
    }
    assert ("prebuild", 0, 64, config.mhc_sinkhorn_iterations, False) in kernels.calls


def test_the_cake_head_prebuilds_the_head_kernels(kernels):
    config = _transformer_config()
    head = CakeMeanOutputContraction(config)

    assert head.num_streams == 4
    assert kernels.calls == [("prebuild", 0, 64, config.mhc_sinkhorn_iterations, True)]


def test_the_cake_module_runs_four_streams_only(kernels):
    with pytest.raises(ValueError, match="runs 4 residual streams, not 2"):
        CakeHyperConnection(_transformer_config(streams=2), 1)


def test_the_cake_module_refuses_the_calls_it_does_not_run(kernels):
    module = CakeHyperConnection(_transformer_config(), 1)
    hidden_states = torch.zeros(3, 2, 4 * 64, dtype=torch.bfloat16)
    branch = torch.zeros(3, 2, 64, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="mHC recompute arena"):
        module(hidden_states, mhc_recompute_manager=object())
    with pytest.raises(NotImplementedError, match="CUDA-graph output slots"):
        module(hidden_states, output_slot=object())
    bda = (None, hidden_states, None)
    with pytest.raises(NotImplementedError, match="without a separate bias"):
        module.fused_h_res_h_post_bda(*bda, (branch, torch.zeros(64)), 0.0, True, False)
    with pytest.raises(NotImplementedError, match="without hidden dropout"):
        module.fused_h_res_h_post_bda(*bda, (branch, None), 0.1, True, False)
    with pytest.raises(NotImplementedError, match="mHC recompute arena"):
        module.fused_h_res_h_post_bda(*bda, (branch, None), 0.0, True, False, manager=object())


def test_a_misaligned_parameter_view_is_refused_not_copied(kernels):
    module = CakeHyperConnection(_transformer_config(), 1)
    weight = module.mapping_proj.weight
    buffer = torch.zeros(weight.numel() + 8, dtype=weight.dtype)
    offset = next(k for k in range(1, 8) if buffer[k:].data_ptr() % 32)
    weight.data = buffer[offset : offset + weight.numel()].view_as(weight)

    with pytest.raises(ValueError, match="mapping_proj.weight in place and needs it 32-byte aligned"):
        module(torch.zeros(3, 2, 4 * 64, dtype=torch.bfloat16))


def _validated(**overrides) -> SimpleNamespace:
    """The namespace fields validate_mhc_backend reads, valid for --mhc-backend cake on GLM-5.3."""
    values = {
        "mhc_backend": "cake",
        "train_backend": "megatron",
        "spec": GLM_SPEC,
        "recompute_modules": ["moe", "mlp"],
        "cuda_graph_impl": "none",
        "use_fused_mhc": False,
        "fp32_residual_connection": False,
        "hidden_dropout": 0.0,
        "overlap_moe_expert_parallel_comm": False,
    }
    return SimpleNamespace(**{**values, **overrides})


@pytest.mark.parametrize("spec", [GLM_SPEC, DSV4_SPEC])
def test_the_supported_specs_validate(spec):
    validate_mhc_backend(_validated(spec=spec))
    validate_mhc_backend(_validated(spec=spec, recompute_modules=None))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("train_backend", "fsdp", "needs --train-backend megatron"),
        ("spec", ["miles_plugins.models.deepseek_v4_1.deepseek_v4_1", "get_dsv41_spec"], "supports --spec"),
        ("spec", ["miles_plugins.models.qwen3_8_next.qwen3_8_next", "get_qwen3_8_next_spec"], "supports --spec"),
        ("spec", None, r"got --spec \(none\)"),
        ("recompute_modules", ["mhc", "mlp"], "drop 'mhc' from --recompute-modules"),
        ("cuda_graph_impl", "transformer_engine", "without CUDA graphs"),
        ("use_fused_mhc", True, "--use-fused-mhc"),
        ("fp32_residual_connection", True, "--fp32-residual-connection"),
        ("hidden_dropout", 0.1, "--hidden-dropout 0"),
        ("overlap_moe_expert_parallel_comm", True, "--overlap-moe-expert-parallel-comm"),
    ],
)
def test_startup_validation_refuses_what_the_cake_module_does_not_run(field, value, match):
    with pytest.raises(ValueError, match=match):
        validate_mhc_backend(_validated(**{field: value}))
    # the same settings are Megatron's business under the default backend
    validate_mhc_backend(_validated(mhc_backend="megatron", **{field: value}))
