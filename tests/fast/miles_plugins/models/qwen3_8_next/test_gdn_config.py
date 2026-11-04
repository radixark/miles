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
        cake = ModuleType("miles_ext_kernels.gdn")
        cake.chunk_gated_delta_rule = cake_kernel
        cake.configure = configured.append
        monkeypatch.setitem(sys.modules, "miles_ext_kernels", ModuleType("miles_ext_kernels"))
        monkeypatch.setitem(sys.modules, "miles_ext_kernels.gdn", cake)
        monkeypatch.setattr(gdn, "ShortConvolution", _CpuSdkModule, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _CpuSdkModule, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        args = (
            None
            if backend is None
            else SimpleNamespace(
                linear_attention_backend=backend, miles_ext_kernels_args={"cake_gdn_recompute_state": True}
            )
        )

        layer = gdn.ReplicatedGatedDeltaNet(config=_config(), layer_idx=0, args=args)
        result = layer.chunk_gated_delta_rule()

        expected = backend or "fla"
        assert calls == [expected]
        assert result == f"{expected}-result"
        expected_configured = [argparse.Namespace(cake_gdn_recompute_state=True)] if backend == "cake" else []
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

        for name in ("fla.ops.gated_delta_rule", "flash_qla", "miles_ext_kernels.gdn"):
            module = ModuleType(name)
            module.chunk_gated_delta_rule = kernel
            module.configure = lambda args: None
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(sys.modules, "miles_ext_kernels", ModuleType("miles_ext_kernels"))
        monkeypatch.setattr(gdn, "ShortConvolution", _PassThroughConv, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _PassThroughNorm, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 3], dtype=torch.int32))
        monkeypatch.setattr(gdn, "build_gdn_cp_context", lambda module, cu_seqlens, device: cp_context)
        args = SimpleNamespace(linear_attention_backend=backend, miles_ext_kernels_args=None)
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
