from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.megatron_utils.test_dsv4_bridge_runtime import _runtime_args

from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.training_utils.model_companion import ModelCompanion


class _SdkModel(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        return value + 1, None


class TestBridgeWitnessConfig:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_bridge_provider_rejects_witness_and_keeps_the_normal_companion_path(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool, bridge_sdk: Mock
    ) -> None:
        """Bridge witness rejection occurs after real configuration and SDK model construction."""
        calls: list[str] = []
        model = _SdkModel()

        def finalize() -> None:
            calls.append("finalize")

        def provide(**kwargs: Any) -> _SdkModel:
            calls.append("provide")
            assert kwargs == {"pre_process": True, "post_process": True, "vp_stage": 2}
            return model

        provider = SimpleNamespace(finalize=finalize, provide=provide)
        bridge = SimpleNamespace(to_megatron_provider=lambda load_weights: provider)
        bridge_sdk.from_hf_pretrained.return_value = bridge
        group = SimpleNamespace(rank=0)
        monkeypatch.setattr(
            "miles.backends.training_utils.parallel.get_parallel_state",
            lambda: SimpleNamespace(pp=group, tp=group, cp=group, intra_dp=group),
        )
        args = _runtime_args()
        args.custom_model_provider_path = None
        args.megatron_to_hf_mode = "bridge"
        args.hf_checkpoint = "sdk-only-model"
        args.enable_witness = enabled

        construct = get_model_provider_func(args=args, role="actor")
        if enabled:
            with pytest.raises(AssertionError, match="Witness is not supported yet in this mode"):
                construct(vp_stage=2)
            assert not any(isinstance(module, ModelCompanion) for module in model.modules())
        else:
            result = construct(vp_stage=2)
            assert result is model
            assert model.model_companion.chunk_index == 2
            assert model(torch.tensor([2.0])).tolist() == [3.0]

        assert calls == ["finalize", "provide"]
