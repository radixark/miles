import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


class _CpuSdkModule(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class TestGdnConfig:
    @pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
    def test_constructor_connects_the_configured_sdk_kernel_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch, backend: str | None, gdn: ModuleType
    ) -> None:
        """The real GDN constructor selects the requested external kernel or its default."""
        calls: list[str] = []

        def fla_kernel() -> str:
            calls.append("fla")
            return "fla-result"

        def qla_kernel() -> str:
            calls.append("flashqla")
            return "flashqla-result"

        fla = ModuleType("fla.ops.gated_delta_rule")
        fla.chunk_gated_delta_rule = fla_kernel
        qla = ModuleType("flash_qla")
        qla.chunk_gated_delta_rule = qla_kernel
        monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
        monkeypatch.setitem(sys.modules, "flash_qla", qla)
        monkeypatch.setattr(gdn, "ShortConvolution", _CpuSdkModule, raising=False)
        monkeypatch.setattr(gdn, "FusedRMSNormGated", _CpuSdkModule, raising=False)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        config = SimpleNamespace(
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
        args = None if backend is None else SimpleNamespace(linear_attention_backend=backend)

        layer = gdn.ReplicatedGatedDeltaNet(config=config, layer_idx=0, args=args)
        result = layer.chunk_gated_delta_rule()

        expected = backend or "fla"
        assert calls == [expected]
        assert result == f"{expected}-result"
        assert layer.in_proj_qkv.weight.shape == (6, 4)
        assert layer.out_proj.weight.device.type == "cpu"

    def test_explicit_unsupported_backend_is_rejected_before_model_allocation(self, gdn: ModuleType) -> None:
        """An unsupported explicit backend cannot silently fall back to FLA."""
        with pytest.raises(ValueError, match="Unsupported GDN backend: unsupported"):
            gdn.ReplicatedGatedDeltaNet(
                config=None, layer_idx=0, args=SimpleNamespace(linear_attention_backend="unsupported")
            )
