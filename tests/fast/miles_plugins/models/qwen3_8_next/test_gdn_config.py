import argparse
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


class _CpuSdkModule(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class _PassThroughConv(_CpuSdkModule):
    def forward(self, x: torch.Tensor, cu_seqlens: Any = None, cp_context: Any = None) -> tuple[torch.Tensor, None]:
        return x, None


class _PassThroughNorm(_CpuSdkModule):
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x


class _WeightedNorm(_PassThroughNorm):
    def __init__(self, hidden_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        hidden_act="silu",
        output_gate_type="sigmoid",
        rms_norm_eps=1e-5,
        dtype=torch.float32,
    )


def _fake_ext(monkeypatch: pytest.MonkeyPatch, op: str, module: ModuleType, configured: list | None = None) -> None:
    """Stand-ins for ``miles_ext_kernels`` (its ``configure`` records the values), ``miles_ext_kernels.cake`` (its
    ``prebuild`` does nothing) and the op module ``miles_ext_kernels.cake.<op>``."""
    package = ModuleType("miles_ext_kernels")
    package.configure = (configured if configured is not None else []).append
    cake = ModuleType("miles_ext_kernels.cake")
    cake.prebuild = lambda op, **params: []
    package.cake = cake
    setattr(cake, op, module)
    for stub in (package, cake, module):
        monkeypatch.setitem(sys.modules, stub.__name__, stub)


class TestGdnConfig:
    @pytest.mark.parametrize("backend", [None, "fla", "flashqla", "cake"])
    def test_constructor_connects_the_configured_sdk_kernel_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch, backend: str | None, gdn: ModuleType
    ) -> None:
        """The real GDN constructor selects the requested external kernel or its default; the cake kernel package
        is configured with the run's arguments."""
        calls: list[str] = []
        configured: list[object] = []

        def fla_kernel() -> str:
            calls.append("fla")
            return "fla-result"

        def qla_kernel() -> str:
            calls.append("flashqla")
            return "flashqla-result"

        def cake_kernel() -> str:
            calls.append("cake")
            return "cake-result"

        fla = ModuleType("fla.ops.gated_delta_rule")
        fla.chunk_gated_delta_rule = fla_kernel
        qla = ModuleType("flash_qla")
        qla.chunk_gated_delta_rule = qla_kernel
        monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
        monkeypatch.setitem(sys.modules, "flash_qla", qla)
        cake = ModuleType("miles_ext_kernels.cake.gdn")
        cake.chunk_gated_delta_rule = cake_kernel
        _fake_ext(monkeypatch, "gdn", cake, configured)
        monkeypatch.setattr(gdn, "ShortConvolution", _CpuSdkModule, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _CpuSdkModule, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        args = (
            None
            if backend is None
            else SimpleNamespace(
                linear_attention_backend=backend,
                linear_attention_conv_backend="fla",
                linear_attention_output_norm_backend="fla",
                miles_ext_kernels_args={"cake_gdn_recompute_state": True},
            )
        )

        layer = gdn.ReplicatedGatedDeltaNet(config=_config(), layer_idx=0, args=args)
        result = layer.chunk_gated_delta_rule()

        expected = backend or "fla"
        assert calls == [expected]
        assert result == f"{expected}-result"
        expected_configured = [{"cake_gdn_recompute_state": True}] if backend == "cake" else []
        assert configured == expected_configured
        assert layer.in_proj_qkv.weight.shape == (6, 4)
        assert layer.out_proj.weight.device.type == "cpu"

    def test_explicit_unsupported_backend_is_rejected_before_model_allocation(self, gdn: ModuleType) -> None:
        """An unsupported explicit backend cannot silently fall back to FLA."""
        with pytest.raises(ValueError, match="Unsupported GDN backend: unsupported"):
            gdn.ReplicatedGatedDeltaNet(
                config=None, layer_idx=0, args=SimpleNamespace(linear_attention_backend="unsupported")
            )

    @pytest.mark.parametrize("backend", ["fla", "cake", "flashqla"])
    def test_context_parallel_runs_the_kernels_that_take_the_cp_context(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, gdn: ModuleType
    ) -> None:
        """Under context parallelism the layer hands fla's CP context to fla or to the cake kernels, which take it
        too, and rejects the others."""
        calls: list[dict[str, Any]] = []

        def kernel(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any):
            calls.append(kwargs)
            return value, None

        for name in ("fla.ops.gated_delta_rule", "flash_qla", "miles_ext_kernels.cake.gdn"):
            module = ModuleType(name)
            module.chunk_gated_delta_rule = kernel
            monkeypatch.setitem(sys.modules, name, module)
        _fake_ext(monkeypatch, "gdn", sys.modules["miles_ext_kernels.cake.gdn"])
        monkeypatch.setattr(gdn, "ShortConvolution", _PassThroughConv, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _PassThroughNorm, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 3], dtype=torch.int32))
        monkeypatch.setattr(gdn, "build_gdn_cp_context", lambda module, cu_seqlens, device: cp_context)
        args = SimpleNamespace(
            linear_attention_backend=backend,
            linear_attention_conv_backend="fla",
            linear_attention_output_norm_backend="fla",
            miles_ext_kernels_args=None,
        )
        layer = gdn.ReplicatedGatedDeltaNet(config=_config(), layer_idx=0, args=args)
        hidden_states = torch.randn(1, 3, 4)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

        if backend == "flashqla":
            with pytest.raises(NotImplementedError, match="requires the 'fla' or 'cake' backend, got 'flashqla'"):
                layer(hidden_states, cu_seqlens=cu_seqlens)
            assert calls == []
            return

        output = layer(hidden_states, cu_seqlens=cu_seqlens)

        assert output.shape == (1, 3, 4)
        assert len(calls) == 1
        assert calls[0]["cp_context"] is cp_context
        assert calls[0]["cu_seqlens"] is cp_context.cu_seqlens

    def test_a_cake_output_norm_is_checked_built_and_run(
        self, monkeypatch: pytest.MonkeyPatch, gdn: ModuleType
    ) -> None:
        """--linear-attention-output-norm-backend cake: the kernel package is configured with the run's values, checks
        the norm (the real package refuses this config's sigmoid gate), builds its kernels and runs the norm."""
        calls: list[tuple] = []
        norm_ext = ModuleType("miles_ext_kernels.gated_rmsnorm")
        norm_ext.configure = lambda namespace: calls.append(("configure", namespace))
        norm_ext.check_supported = lambda *args, **kwargs: calls.append(("check", args, kwargs))
        norm_ext.prebuild = lambda: calls.append(("prebuild",))

        def rmsnorm_silu_gate(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
            calls.append(("run", tuple(x.shape), tuple(z.shape), eps))
            return x

        norm_ext.rmsnorm_silu_gate = rmsnorm_silu_gate
        fla = ModuleType("fla.ops.gated_delta_rule")
        fla.chunk_gated_delta_rule = lambda query, key, value, **kwargs: (value, None)
        monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
        monkeypatch.setitem(sys.modules, "miles_ext_kernels", ModuleType("miles_ext_kernels"))
        monkeypatch.setitem(sys.modules, "miles_ext_kernels.gated_rmsnorm", norm_ext)
        monkeypatch.setattr(gdn, "ShortConvolution", _PassThroughConv, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _WeightedNorm, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        args = SimpleNamespace(
            linear_attention_backend="fla",
            linear_attention_conv_backend="fla",
            linear_attention_output_norm_backend="cake",
            miles_ext_kernels_args={"cake_gated_rmsnorm_variant": "det"},
        )

        layer = gdn.ReplicatedGatedDeltaNet(config=_config(), layer_idx=0, args=args)
        output = layer(torch.randn(1, 3, 4), cu_seqlens=torch.tensor([0, 3], dtype=torch.int32))

        assert calls == [
            ("configure", argparse.Namespace(cake_gated_rmsnorm_variant="det")),
            ("check", (2, "sigmoid", torch.float32), {"has_bias": False}),
            ("prebuild",),
            ("run", (3, 2), (3, 2), 1e-5),
        ]
        assert output.shape == (1, 3, 4)
