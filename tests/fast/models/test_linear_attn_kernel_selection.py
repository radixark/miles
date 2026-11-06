"""Linear-attention kernel selection, and fla vs flashqla numerical equivalence for GDN.

The flashqla equivalence tests need a Hopper (SM90+) GPU with both `fla` and `flash_qla`; they skip otherwise.
The cake kernels' equivalence with fla and their determinism are tested in the miles-ext-kernels submodule
(`3rdparty/miles-ext-kernels/tests/cake/gdn`), which `tests/fast-gpu/kernels/test_miles_ext_kernels.py` runs on
Blackwell.
The gradients of the reproducible gated output norm of the cake layers are checked on the CPU against autograd.
"""

import os
import sys
import types

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.trainer_utils import compute_trainer_config
from miles_plugins.models import linear_attn


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        linear_attn.gdn_kernel("nope")


def test_cake_backend_routes_to_the_generated_kernels():
    pytest.importorskip("torch")
    pytest.importorskip("miles_ext_kernels.cake.gdn", reason="the miles-ext-kernels submodule is not checked out")
    fn = linear_attn.gdn_kernel("cake")
    assert fn.__module__ == "miles_ext_kernels.cake._impl.gdn.ops"
    assert fn.__name__ == "chunk_gated_delta_rule"


def _fake_cake_package(monkeypatch, configured: list, prebuilt: list | None = None) -> None:
    """Stand-ins for ``miles_ext_kernels`` (its ``configure`` records the values), ``miles_ext_kernels.cake`` (its
    ``prebuild`` records the op) and the op module ``miles_ext_kernels.cake.gdn``."""
    package = types.ModuleType("miles_ext_kernels")
    package.configure = configured.append
    cake = types.ModuleType("miles_ext_kernels.cake")
    cake.prebuild = lambda op, **params: (prebuilt if prebuilt is not None else []).append((op, params))
    gdn = types.ModuleType("miles_ext_kernels.cake.gdn")
    gdn.chunk_gated_delta_rule = "cake-kernel"
    package.cake, cake.gdn = cake, gdn
    for module in (package, cake, gdn):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def test_cake_backend_hands_the_kernel_package_its_own_argument_values(monkeypatch):
    """miles passes the values of the kernel package's own arguments to ``miles_ext_kernels.configure`` without
    knowing their names, then builds the op's libraries, where a layer is built (the caller has ``args``)."""
    configured, prebuilt = [], []
    _fake_cake_package(monkeypatch, configured, prebuilt)
    fla = types.ModuleType("fla.ops.gated_delta_rule")
    fla.chunk_gated_delta_rule = "fla-kernel"
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
    args = types.SimpleNamespace(linear_attention_backend="cake", miles_ext_kernels_args={"kernel_option": 3})

    assert linear_attn.gdn_kernel("cake") == "cake-kernel"
    assert linear_attn.gdn_kernel("fla", args=args) == "fla-kernel"
    assert configured == prebuilt == []
    assert linear_attn.gdn_kernel("cake", args=args) == "cake-kernel"
    assert configured == [{"kernel_option": 3}]
    assert prebuilt == [("gdn", {})]
    # a configuration without the flag (no ext arguments) configures the package with its defaults
    assert linear_attn.gdn_kernel("cake", args=types.SimpleNamespace(miles_ext_kernels_args=None)) == "cake-kernel"
    assert configured[-1] is None


@pytest.mark.parametrize(
    "extra, variant, recompute",
    [
        ([], "fast", False),
        (["--cake-gdn-recompute-state"], "fast", True),
        (["--cake-gdn-variant", "det"], "det", False),
        (["--cake-gdn-variant", "fast", "--cake-gdn-recompute-state"], "fast", True),
    ],
)
def test_the_parsed_ext_arguments_reach_the_cake_kernel_package_through_the_trainer_config(
    monkeypatch, extra, variant, recompute
):
    """The real path: the submodule's GDN arguments, parsed and validated by miles (an unset variant resolves to
    fast outside deterministic mode), sliced into the trainer's typed configuration and handed by the layer seam to
    ``miles_ext_kernels.configure``."""
    config = parse_megatron_test_config("--use-miles-ext-kernels", "--linear-attention-backend", "cake", *extra)
    trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])
    configured = []
    _fake_cake_package(monkeypatch, configured)

    linear_attn.gdn_kernel(trainer.linear_attention_backend, args=trainer)

    gdn_args = {name: value for name, value in trainer.miles_ext_kernels_args.items() if name.startswith("cake_gdn_")}
    assert gdn_args == {"cake_gdn_variant": variant, "cake_gdn_recompute_state": recompute}
    assert configured == [trainer.miles_ext_kernels_args]


def test_the_gdn_variant_needs_the_cake_backend():
    with pytest.raises(ValueError, match="--cake-gdn-variant requires --linear-attention-backend cake"):
        parse_megatron_test_config("--use-miles-ext-kernels", "--cake-gdn-variant", "det")


def test_cake_backend_without_the_submodule_names_the_flag_and_the_checkout(monkeypatch):
    monkeypatch.setitem(sys.modules, "miles_ext_kernels", None)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake", None)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake.gdn", None)

    with pytest.raises(ImportError, match="--use-miles-ext-kernels") as excinfo:
        linear_attn.gdn_kernel("cake")
    assert "git submodule update --init 3rdparty/miles-ext-kernels" in str(excinfo.value)


@pytest.mark.parametrize(
    "capability,user_value,expected", [((10, 0), None, "0"), ((9, 0), None, None), ((10, 0), "1", "1")]
)
def test_kda_uses_the_triton_backward_on_blackwell_unless_set(monkeypatch, capability, user_value, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "fla.ops.kda", types.SimpleNamespace(chunk_kda="chunk_kda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    if user_value is not None:
        monkeypatch.setenv("FLA_TILELANG", user_value)
    assert linear_attn.kda_kernel.__wrapped__() == "chunk_kda"
    assert os.environ.get("FLA_TILELANG") == expected


def test_cake_kda_backend_routes_to_the_kernel_package(monkeypatch):
    linear_attn.kda_kernel.cache_clear()
    cake = types.ModuleType("miles_ext_kernels.cake.kda")
    cake.chunk_kda = "cake-kda"
    monkeypatch.setitem(sys.modules, "miles_ext_kernels", types.ModuleType("miles_ext_kernels"))
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake", types.ModuleType("miles_ext_kernels.cake"))
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake.kda", cake)
    try:
        assert linear_attn.kda_kernel("cake") == "cake-kda"
    finally:
        linear_attn.kda_kernel.cache_clear()


def test_unknown_kda_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported KDA backend"):
        linear_attn.kda_kernel.__wrapped__("nope")


def test_cake_kda_backend_without_the_submodule_names_the_flag(monkeypatch):
    monkeypatch.setitem(sys.modules, "miles_ext_kernels", None)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake", None)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.cake.kda", None)

    with pytest.raises(ImportError, match="--use-miles-ext-kernels") as excinfo:
        linear_attn.kda_kernel.__wrapped__("cake")
    assert "git submodule update --init 3rdparty/miles-ext-kernels" in str(excinfo.value)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--context-parallel-size", "2"], "--kda-backend cake does not support context parallelism"),
        (["--qkv-format", "bshd"], "--kda-backend cake needs --qkv-format thd"),
    ],
)
def test_the_cake_kda_backend_rejects_what_its_kernels_do_not_run(extra, message):
    with pytest.raises(ValueError, match=message):
        parse_megatron_test_config("--use-miles-ext-kernels", "--kda-backend", "cake", *extra)


def test_the_cake_kda_backend_parses_with_its_supported_layout():
    config = parse_megatron_test_config("--use-miles-ext-kernels", "--kda-backend", "cake")

    assert config.kda_backend == "cake"


def test_short_conv_backend_respects_fla_conv_backend(monkeypatch):
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    assert linear_attn.short_conv_backend.__wrapped__() == "cuda"


def test_cake_short_conv_defaults_to_triton_unless_set(monkeypatch):
    monkeypatch.delenv("FLA_CONV_BACKEND", raising=False)
    assert linear_attn.cake_conv_backend.__wrapped__() == "triton"
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    assert linear_attn.cake_conv_backend.__wrapped__() == "cuda"


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_the_reproducible_gated_norm_matches_autograd_of_its_formula(activation):
    """The hand-written backward of ``reproducible_rms_norm_gated`` against autograd through the same formula in
    FP64 (``rms_norm(x) * weight * act(g)``)."""
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    x, g, dy = (torch.randn(37, 128, generator=gen) for _ in range(3))
    weight = 1 + 0.1 * torch.randn(128, generator=gen)
    leaves = [t.clone().requires_grad_() for t in (x, g, weight)]
    out = linear_attn.reproducible_rms_norm_gated(*leaves, None, activation, 1e-6)
    out.backward(dy)

    ref = [t.double().requires_grad_() for t in (x, g, weight)]
    gate = ref[1].sigmoid() if activation == "sigmoid" else torch.nn.functional.silu(ref[1])
    expected = ref[0] * torch.rsqrt(ref[0].square().mean(-1, keepdim=True) + 1e-6) * ref[2] * gate
    expected.backward(dy.double())
    torch.testing.assert_close(out.double(), expected.detach(), rtol=1e-5, atol=1e-6)
    for got, want in zip(leaves, ref, strict=True):
        torch.testing.assert_close(got.grad.double(), want.grad, rtol=1e-4, atol=1e-5)


def test_the_reproducible_gated_norm_refuses_a_bias_and_other_gates():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 128)
    with pytest.raises(ValueError, match="no bias"):
        linear_attn.reproducible_rms_norm_gated(x, x, torch.ones(128), torch.zeros(128), "silu", 1e-6)
    with pytest.raises(ValueError, match="gate activation"):
        linear_attn.reproducible_rms_norm_gated(x, x, torch.ones(128), None, "relu", 1e-6)


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

    fla_kernel = linear_attn.gdn_kernel("fla")
    flashqla_kernel = linear_attn.gdn_kernel("flashqla")

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
